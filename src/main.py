import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
import argparse
import yaml
import threading
import queue
import concurrent.futures
from google.cloud import storage

# Import custom modules
from speed_detection_module.camera_calibration import CameraCalibration
from speed_detection_module.car_detection import CarDetection
from speed_detection_module.bounding_box_constructor import BoundingBoxConstructor
from speed_detection_module.vehicle_tracker import VehicleTracker
from speed_detection_module.depth_estimation import DepthEstimationModel
from speed_detection_module.masker import Masker
from speed_detection_module.speed_estimator import SpeedEstimator
from anpr_module.Help_util import assign_car, preprocess_frame, read_license_plate, write_csv

from ultralytics import YOLO


class IntegratedVideoProcessor:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Set up Google Cloud credentials
        self.setup_google_cloud_credentials()

        # Initialize variables from config
        self.video_path = self.config['video_path']
        self.calibration_file = self.config['calibration_file']
        self.road_mask_file = self.config['road_mask_file']
        self.detection_confidence = self.config['detection_confidence']
        self.speed_threshold = self.config['speed_threshold']
        self.gcs_bucket_name = self.config['gcs_bucket_name']
        self.frame_skip = self.config['frame_skip']

        # Initialize YOLO model for license plate detection
        try:
            self.license_plate_detector = YOLO(self.config['LP_model'])
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            raise

        # Initialize other components
        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.meters_per_pixel = self.config.get('meters_per_pixel', 0.1)
        self.speed_estimator = None  # Will be initialized after camera calibration

        # Initialize Google Cloud Storage
        self.initialize_google_cloud_storage()

        # Video capture setup
        self.cap = cv2.VideoCapture(self.video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.results = []
        self.license_plate_results = {}
        self.last_saved_frame = 0
        self.save_interval = 100  # Save results every 100 frames

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output_video.mp4', fourcc, self.fps, (self.width, self.height))

        # Threading setup
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()

    def setup_google_cloud_credentials(self):
        """Set up Google Cloud credentials."""
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            self.logger.info("Google Cloud credentials already set in environment.")
            return

        cred_path = self.config.get('google_cloud_credentials')
        if cred_path and os.path.exists(cred_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
            self.logger.info(f"Set Google Cloud credentials from config: {cred_path}")
        else:
            self.logger.warning("Google Cloud credentials not found. Some features may not work.")

    def initialize_google_cloud_storage(self):
        """Initialize Google Cloud Storage client."""
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
            self.logger.info("Successfully initialized Google Cloud Storage client.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Cloud Storage: {str(e)}")
            self.storage_client = None
            self.bucket = None

    def save_image_to_gcs(self, image, gcs_subfolder, filename):
        """Saves an image directly to GCS."""
        if self.bucket is None:
            self.logger.warning("GCS not initialized. Skipping image upload.")
            return

        try:
            gcs_path = f"{gcs_subfolder}/{filename}"
            _, encoded_image = cv2.imencode('.jpg', image)
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')
            self.logger.info(f'Image saved to {gcs_path}')
        except Exception as e:
            self.logger.error(f"Failed to save image to GCS: {str(e)}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), color, 2)

        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])
            draw_line(corners[i + 4], corners[(i + 1) % 4 + 4])
            draw_line(corners[i], corners[i + 4])

        return img

    def save_results_to_gcs(self, frame_count=None):
        """Saves results to Google Cloud Storage."""
        if self.bucket is None:
            self.logger.warning("GCS not initialized. Skipping results upload.")
            return

        if frame_count is None:
            results_to_save = self.results
            blob_name = f'output/speed_estimation_results_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        else:
            results_to_save = [r for r in self.results if r['frame'] == frame_count]
            blob_name = f'output/speed_estimation_results_frame_{frame_count}.json'

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        serializable_results = json.loads(json.dumps(results_to_save, default=convert_to_serializable))
        results_json = json.dumps(serializable_results, indent=2)

        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(results_json, content_type='application/json')
        self.logger.info(f'Results JSON uploaded to GCS: {blob_name}')

    def process_frame(self, frame_data):
        frame, frame_count, current_time = frame_data
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        vis_frame = frame.copy()

        # Vehicle Detection and Speed Calculation
        masked_frame = self.masker.apply_mask(frame)
        ipm_frame = self.calibration.apply_ipm(masked_frame)
        depth_map = self.depth_model.estimate_depth(ipm_frame)

        vehicle_detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)
        self.logger.info(f"Frame {frame_count}: Received {len(vehicle_detections)} detections")

        bboxes_3d = []
        for det in vehicle_detections:
            x1, y1, x2, y2, conf, cls = det
            center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
            if np.isnan(center_depth) or np.isinf(center_depth):
                self.logger.warning(f"Invalid depth value for detection {det}")
                continue
            bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
            if bbox_3d is not None:
                bboxes_3d.append(bbox_3d)

        # Tracking and Speed Calculation
        tracks = self.tracker.update(bboxes_3d)
        self.logger.info(f"Frame {frame_count}: Tracking {len(tracks)} vehicles")

        results = []
        for track_id, track in tracks.items():
            corners_3d = track['bbox_3d']
            corners_2d = self.bbox_constructor.project_3d_to_2d(corners_3d)

            self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))

            current_position = np.mean(corners_3d, axis=0)
            ipm_position = self.calibration.apply_ipm(np.array([current_position[:2]]))
            estimated_speed = self.speed_estimator.estimate_speed(track_id, ipm_position[0], current_time, frame_count)

            self.logger.info(
                f"Track {track_id}: position = {current_position.T}, estimated speed = {estimated_speed:.2f} km/h")

            speed_alert = estimated_speed > self.speed_threshold if estimated_speed is not None else False

            if corners_2d.shape[0] >= 4:
                x1, y1 = np.min(corners_2d, axis=0).astype(int)
                x2, y2 = np.max(corners_2d, axis=0).astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                speed_text = f"ID: {track_id}, Speed: {estimated_speed:.2f} km/h" if estimated_speed is not None else f"ID: {track_id}, Speed: N/A"
                cv2.putText(vis_frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if speed_alert:
                license_plate_detections = self.license_plate_detector(frame)[0]
                for license_plate in license_plate_detections.boxes.data.tolist():
                    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, _ = license_plate
                    x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicle_detections)
                    if car_id != -1:
                        license_plate_image = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp)]
                        license_plate_image_gcs_path = f'Images/license_plates/license_plate_{car_id}_{timestamp}.jpg'
                        self.save_image_to_gcs(license_plate_image, 'Images/license_plates',
                                               f'license_plate_{car_id}_{timestamp}.jpg')

                        lp_crop = preprocess_frame(frame, x1_lp, y1_lp, x2_lp, y2_lp)
                        license_plate_text, lp_confidence = read_license_plate(lp_crop)

                        self.license_plate_results[frame_count] = {
                            'track_id': track_id,
                            'license_plate': {
                                'text': license_plate_text,
                                'confidence': lp_confidence,
                                'bbox': [x1_lp, y1_lp, x2_lp, y2_lp]
                            },
                            'license_plate_image_gcs_path': license_plate_image_gcs_path,
                            'speed': estimated_speed,
                            'timestamp': timestamp,
                            'speed_alert': speed_alert
                        }

            result = {
                'frame': frame_count,
                'track_id': track_id,
                'speed': estimated_speed if estimated_speed is not None else 'N/A',
                'license_plate': self.license_plate_results.get(frame_count, {}).get('license_plate', 'N/A'),
                'timestamp': timestamp,
                'speed_alert': speed_alert
            }
            results.append(result)

        self.results.extend(results)
        self.out.write(vis_frame)
        self.save_results_to_gcs(frame_count)

        return results

    def process_video(self):
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            self.logger.info(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            self.logger.info("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            self.logger.info(f"Saved road mask to {self.road_mask_file}")

        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            self.logger.info(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            self.logger.info("Performing camera calibration...")
            calibration_frames = self._collect_calibration_frames()
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            self.logger.info(f"Saved camera calibration to {self.calibration_file}")

        dummy_frame = np.zeros((self.height, self.width), dtype=np.uint8)
        ipm_frame = self.calibration.apply_ipm(dummy_frame)
        ipm_height, ipm_width = ipm_frame.shape[:2]

        self.logger.info("Please select two lines for speed calculation on the IPM view...")
        lines = self.select_speed_lines(ipm_frame)
        if lines is None:
            self.logger.warning("Line selection cancelled. Using default values.")
            line1_y = int(ipm_height * 0.2)
            line2_y = int(ipm_height * 0.8)
        else:
            line1_y, line2_y = lines

        self.speed_estimator = SpeedEstimator(ipm_height, ipm_width, self.fps, self.meters_per_pixel, line1_y, line2_y)
        self.logger.info(f"SpeedEstimator initialized with IPM dimensions: {ipm_width}x{ipm_height}")
        self.logger.info(f"Speed calculation lines set at y={line1_y} and y={line2_y}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.logger.info("Video capture reset to beginning.")

        self.ipm_matrix = self.calibration.ipm_matrix
        self.camera_matrix = self.calibration.get_camera_matrix()
        self.vanishing_points = self.calibration.vanishing_points

        self.bbox_constructor = BoundingBoxConstructor(self.vanishing_points, self.camera_matrix)

        frame_count = -1

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_frame = {}
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % self.frame_skip != 0:
                    continue

                current_time = frame_count / self.fps

                future = executor.submit(self.process_frame, (frame, frame_count, current_time))
                future_to_frame[future] = frame_count

            for future in concurrent.futures.as_completed(future_to_frame):
                frame_count = future_to_frame[future]
                try:
                    result = future.result()
                    self.logger.info(f"Processed frame {frame_count}")
                except Exception as exc:
                    self.logger.error(f"Error processing frame {frame_count}: {exc}")

        self.save_results_to_gcs()
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()
        self.upload_video_to_gcs('output_video.mp4')

    def _collect_calibration_frames(self, num_frames=10):
        calibration_frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                calibration_frames.append(frame)
            else:
                break
        return calibration_frames

    def select_speed_lines(self, frame):
        lines = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(lines) < 2:
                    lines.append(y)
                    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
                    cv2.imshow('Select Speed Lines', frame)

                if len(lines) == 2:
                    cv2.setMouseCallback('Select Speed Lines', lambda *args: None)

        clone = frame.copy()
        cv2.namedWindow('Select Speed Lines')
        cv2.setMouseCallback('Select Speed Lines', mouse_callback)

        while len(lines) < 2:
            cv2.imshow('Select Speed Lines', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                frame = clone.copy()
                lines = []
            elif key == 27:  # ESC key
                break

        cv2.destroyWindow('Select Speed Lines')

        if len(lines) == 2:
            return sorted(lines)
        else:
            return None

    def upload_video_to_gcs(self, video_path):
        """Uploads the processed video to Google Cloud Storage."""
        if self.bucket is None:
            self.logger.warning("GCS not initialized. Skipping video upload.")
            return

        try:
            blob = self.bucket.blob(f'processed_videos/{os.path.basename(video_path)}')
            blob.upload_from_filename(video_path)
            self.logger.info(f'Processed video uploaded to GCS: {blob.public_url}')
        except Exception as e:
            self.logger.error(f"Failed to upload video to GCS: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Integrated Video Processor for Speed Detection and ANPR")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file (default: config/config.yaml)')
    args = parser.parse_args()
    config_path = args.config

    try:
        processor = IntegratedVideoProcessor(args.config)
        processor.process_video()
    except Exception as e:
        logging.error(f"An error occurred during video processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
