#!/usr/bin/env python3
"""
Computer Vision Module for CARLA Data Recorder
Optimized YOLOv11m-obb vehicle detection for CARLA simulation
Enhanced with VehicleDetectionTracker features: speed calculation, direction mapping, color/model classification
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
import time
import math
import json
import base64
from collections import defaultdict
from datetime import datetime

# YOLOv11 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ Ultralytics YOLO imported successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Ultralytics not found. Install with: pip install ultralytics")

# Torch imports for GPU optimization
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Color classifier imports
try:
    from github.color_classifier.classifier import Classifier as ColorClassifier
    COLOR_CLASSIFIER_AVAILABLE = True
    print("✅ Color classifier available")
except ImportError:
    COLOR_CLASSIFIER_AVAILABLE = False
    print("⚠️ Color classifier not available")

# Model classifier imports
try:
    from github.model_classifier.classifier import Classifier as ModelClassifier
    MODEL_CLASSIFIER_AVAILABLE = True
    print("✅ Model classifier available")
except ImportError:
    MODEL_CLASSIFIER_AVAILABLE = False
    print("⚠️ Model classifier not available")

class ComputerVisionProcessor:
    """
    CARLA-optimized computer vision processor for vehicle detection.
    Uses YOLOv11m-obb for oriented bounding box detection of vehicles.
    """
    
    def __init__(self):
        """Initialize the computer vision processor."""
        print("🔍 Initializing Computer Vision Processor...")
        # Detection statistics
        self.detection_stats = {
            "total_vehicles": 0,
            "close_vehicles": 0,
            "medium_vehicles": 0,
            "far_vehicles": 0,
            "processing_time": 0.0
        }
        # Distance thresholds for classification (in pixels)
        self.close_threshold = 150
        self.medium_threshold = 300

        # --- Enhanced direction tracking additions ---
        self.prev_detections = {}  # {track_id: {'position': (x, y), 'time': timestamp, 'cls': class_id}}
        self.vehicle_directions = {}  # {track_id: {'vector': (dx, dy), 'magnitude': float, 'angle': float, 'smoothed_vector': (dx, dy)}}
        self.track_history = {}  # {track_id: [positions]}
        self.max_track_history = 15  # Increased for better direction estimation
        self.max_track_age = 2.0  # Increased to 2 seconds for better persistence
        
        # Enhanced frame management for adaptive performance
        self.frame_skip_counter = 0
        self.base_frame_skip_rate = 2  # Base frame skip rate
        self.frame_skip_rate = 2  # Current frame skip rate (starts at base rate)
        self.adaptive_frame_skip = True  # Enable adaptive frame skipping
        self.processing_times = []  # Track processing performance
        self.max_processing_time = 0.1  # 100ms target
        self.last_processed_frame = None
        
        # Enhanced tracking parameters for improved matching logic
        self.tracking_confidence_threshold = 0.7  # Minimum confidence for new tracks
        self.position_variance_threshold = 5.0  # Maximum position variance for track consistency
        self.direction_consistency_weight = 0.3  # Weight for direction consistency in matching
        
        # Vehicle type mapping for visual clarity
        self.vehicle_type_names = {
            1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 
            7: "truck", 9: "boat", 10: "traffic_light"
        }
        # --- End enhanced direction tracking additions ---

        # === NEW: VehicleDetectionTracker Features ===
        # ByteTrack integration for persistent tracking
        self.use_bytetrack = True
        self.tracker_config = "bytetrack.yaml"
        
        # Speed and direction calculation
        self.vehicle_timestamps = defaultdict(lambda: {"timestamps": [], "positions": []})
        self.vehicle_speeds = {}  # {track_id: speed_info}
        self.direction_ranges = {
            (-math.pi / 8, math.pi / 8): "Right",
            (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
            (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
            (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
            (7 * math.pi / 8, -7 * math.pi / 8): "Left",
            (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
            (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
            (-3 * math.pi / 8, -math.pi / 8): "Top Right"
        }
        
        # Vehicle classification
        self.color_classifier = None
        self.model_classifier = None
        self.enable_color_classification = COLOR_CLASSIFIER_AVAILABLE
        self.enable_model_classification = MODEL_CLASSIFIER_AVAILABLE
        
        # Vehicle data storage
        self.detected_vehicles = set()
        self.vehicle_frames = {}  # Store individual vehicle frames
        self.vehicle_color_info = {}  # Store color classification results
        self.vehicle_model_info = {}  # Store model classification results
        
        # Performance tracking for ByteTrack
        self.max_track_history_bytetrack = 30  # Track history for ByteTrack
        self.track_thickness = 2
        
        # Brightness enhancement for better detection
        self.brightness_factor = 1.5
        # === END: VehicleDetectionTracker Features ===

        # Initialize YOLO model
        if YOLO_AVAILABLE:
            print("🚀 Loading YOLOv11m-obb model...")
            try:
                self.yolo_model = YOLO("yolo11m-obb.pt")
                # Force CPU inference to avoid GPU OOM issues
                import torch
                self.yolo_model.model.cpu()
                for param in self.yolo_model.model.parameters():
                    param.requires_grad = False
                print("✅ YOLOv11m-obb loaded successfully (CPU-only mode)")
            except Exception as e:
                print(f"❌ Failed to load YOLOv11m-obb: {e}")
                print("⚠️ Attempting to load YOLOv8n as fallback...")
                try:
                    self.yolo_model = YOLO("yolov8n.pt")
                    # Force CPU inference for fallback too
                    import torch
                    self.yolo_model.model.cpu()
                    for param in self.yolo_model.model.parameters():
                        param.requires_grad = False
                    print("✅ YOLOv8n loaded as fallback (CPU-only mode)")
                except:
                    self.yolo_model = None
                    print("❌ Failed to load any YOLO model")
        else:
            self.yolo_model = None
            print("❌ YOLO not available. Vehicle detection disabled.")

        # Bind enhanced match detections method
        self._match_detections = self.enhanced_match_detections
        
        # Initialize classifiers
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize color and model classifiers if available."""
        if self.enable_color_classification and self.color_classifier is None:
            try:
                self.color_classifier = ColorClassifier()
                print("✅ Color classifier initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize color classifier: {e}")
                self.enable_color_classification = False
        
        if self.enable_model_classification and self.model_classifier is None:
            try:
                self.model_classifier = ModelClassifier()
                print("✅ Model classifier initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize model classifier: {e}")
                self.enable_model_classification = False
    
    def _map_direction_to_label(self, direction):
        """
        Map direction angle to human-readable label.
        Args:
            direction (float): Direction angle in radians
        Returns:
            str: Direction label (e.g., "Right", "Top Left", etc.)
        """
        for angle_range, label in self.direction_ranges.items():
            if angle_range[0] <= direction <= angle_range[1]:
                return label
        return "Unknown"
    
    def _encode_image_base64(self, image):
        """
        Encode an image as base64.
        Args:
            image (numpy.ndarray): The image to be encoded.
        Returns:
            str: Base64-encoded image.
        """
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    
    def _decode_image_base64(self, image_base64):
        """
        Decode a base64-encoded image.
        Args:
            image_base64 (str): Base64-encoded image data.
        Returns:
            numpy.ndarray or None: Decoded image as a numpy array or None if decoding fails.
        """
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"⚠️ Failed to decode base64 image: {e}")
            return None
    
    def _increase_brightness(self, image, factor=None):
        """
        Increases the brightness of an image by multiplying its pixels by a factor.
        Args:
            image: The input image in numpy array format.
            factor: The brightness increase factor. A value greater than 1 will increase brightness.
        Returns:
            The image with increased brightness.
        """
        if factor is None:
            factor = self.brightness_factor
        brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened_image
    
    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        """Convert meters per second to kilometers per hour."""
        return meters_per_second * 3.6
    
    def _calculate_speed_and_direction(self, track_id, current_pos, frame_timestamp):
        """
        Calculate speed and direction for a tracked vehicle.
        Args:
            track_id: Unique identifier for the tracked vehicle
            current_pos: Current position (x, y)
            frame_timestamp: Timestamp of current frame
        Returns:
            dict: Speed and direction information
        """
        # Store timestamp and position
        if track_id not in self.vehicle_timestamps:
            self.vehicle_timestamps[track_id] = {"timestamps": [], "positions": []}
        
        self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
        self.vehicle_timestamps[track_id]["positions"].append(current_pos)
        
        timestamps = self.vehicle_timestamps[track_id]["timestamps"]
        positions = self.vehicle_timestamps[track_id]["positions"]
        
        speed_kph = None
        reliability = 0.0
        direction_label = None
        direction = None
        
        if len(timestamps) >= 2:
            delta_t_list = []
            distance_list = []
            
            # Calculate time intervals and distances
            for i in range(1, len(timestamps)):
                t1, t2 = timestamps[i - 1], timestamps[i]
                if hasattr(t1, 'timestamp') and hasattr(t2, 'timestamp'):
                    delta_t = t2.timestamp() - t1.timestamp()
                else:
                    # Handle case where timestamps are not datetime objects
                    delta_t = float(t2) - float(t1)
                
                if delta_t > 0:
                    x1, y1 = positions[i - 1]
                    x2, y2 = positions[i]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    delta_t_list.append(delta_t)
                    distance_list.append(distance)
            
            # Calculate average speed
            if delta_t_list and distance_list:
                speeds = [distance / delta_t for distance, delta_t in zip(distance_list, delta_t_list)]
                if speeds:
                    avg_speed_mps = sum(speeds) / len(speeds)
                    speed_kph = self._convert_meters_per_second_to_kmph(avg_speed_mps)
            
            # Calculate direction
            initial_x, initial_y = positions[0]
            final_x, final_y = positions[-1]
            direction = math.atan2(final_y - initial_y, final_x - initial_x)
            direction_label = self._map_direction_to_label(direction)
            
            # Calculate reliability
            sample_count = len(timestamps)
            if sample_count < 5:
                reliability = 0.5
            elif sample_count < 10:
                reliability = 0.7
            else:
                reliability = 1.0
        
        return {
            "kph": speed_kph,
            "reliability": reliability,
            "direction_label": direction_label,
            "direction": direction
        }
    
    def _classify_vehicle(self, vehicle_frame, track_id):
        """
        Classify vehicle color and model.
        Args:
            vehicle_frame: Cropped image of the vehicle
            track_id: Unique identifier for the tracked vehicle
        Returns:
            tuple: (color_info, model_info) as JSON strings
        """
        color_info = None
        model_info = None
        
        if self.enable_color_classification and self.color_classifier is not None:
            try:
                color_info = self.color_classifier.predict(vehicle_frame)
                self.vehicle_color_info[track_id] = color_info
            except Exception as e:
                print(f"⚠️ Color classification failed for track {track_id}: {e}")
        
        if self.enable_model_classification and self.model_classifier is not None:
            try:
                model_info = self.model_classifier.predict(vehicle_frame)
                self.vehicle_model_info[track_id] = model_info
            except Exception as e:
                print(f"⚠️ Model classification failed for track {track_id}: {e}")
        
        return (
            json.dumps(color_info) if color_info else None,
            json.dumps(model_info) if model_info else None
        )

    def preprocess_image_for_stationary(self, image):
        """
        Preprocess image to enhance detection of stationary or low-contrast vehicles.
        Args:
            image: Input image frame
        Returns:
            Enhanced image for better YOLO detection
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply subtle sharpening filter
        kernel_sharpening = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpening * 0.1)
        
        # Blend original and processed image for natural look
        result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
        
        return result

    def calculate_track_stability(self, track_id):
        """
        Calculate stability score for a track based on movement consistency.
        Args:
            track_id: ID of the track to analyze
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return 0.5  # Default score for new tracks
        
        positions = self.track_history[track_id]
        if len(positions) < 3:
            return 0.5
        
        # Calculate position variance
        positions_array = np.array(positions)
        variance = np.var(positions_array, axis=0)
        avg_variance = np.mean(variance)
        
        # Normalize variance to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + avg_variance / self.position_variance_threshold)
        
        return min(max(stability, 0.0), 1.0)

    def get_direction_from_history(self, track_id, min_points=3):
        """
        Calculate direction vector from track history using multiple points.
        Args:
            track_id: ID of the track
            min_points: Minimum number of points required
        Returns:
            Direction vector (dx, dy) or None if insufficient data
        """
        if track_id not in self.track_history:
            return None
        
        positions = self.track_history[track_id]
        if len(positions) < min_points:
            return None
        
        # Use linear regression on the last several points for robust direction estimation
        positions_array = np.array(positions[-min_points:])
        if len(positions_array) < 2:
            return None
        
        # Simple linear fit
        x_coords = positions_array[:, 0]
        y_coords = positions_array[:, 1]
        
        if len(x_coords) >= 2:
            # Calculate average direction vector
            dx = x_coords[-1] - x_coords[0]
            dy = y_coords[-1] - y_coords[0]
            
            # Apply temporal weighting (recent positions matter more)
            weights = np.linspace(0.5, 1.0, len(positions_array))
            weighted_dx = np.average(np.diff(x_coords), weights=weights[1:])
            weighted_dy = np.average(np.diff(y_coords), weights=weights[1:])
            
            return (weighted_dx * len(x_coords), weighted_dy * len(y_coords))
        
        return None

    def enhanced_match_detections(self, current_detections):
        """
        Enhanced detection matching with improved proximity thresholding, track history analysis,
        and better direction estimation using multiple points.
        Args:
            current_detections: List of dicts with 'position', 'box', 'conf', 'cls'
        Returns:
            List of detections with 'track_id' assigned and improved direction vectors
        """
        current_time = time.time()
        matched_detections = []
        used_prev_ids = set()

        # Remove old tracks and calculate stability scores
        valid_track_ids = set()
        track_stability_scores = {}
        
        for track_id in list(self.prev_detections.keys()):
            if current_time - self.prev_detections[track_id]['time'] > self.max_track_age:
                # Remove stale tracks
                self.prev_detections.pop(track_id, None)
                self.vehicle_directions.pop(track_id, None)
                self.track_history.pop(track_id, None)
            else:
                valid_track_ids.add(track_id)
                # Calculate stability score for better matching
                track_stability_scores[track_id] = self.calculate_track_stability(track_id)

        # Track id counter (ensures no unbounded growth)
        if valid_track_ids:
            next_track_id = max(valid_track_ids) + 1
        else:
            next_track_id = 0

        # Enhanced matching with multiple criteria
        for detection in current_detections:
            cur_pos = detection['position']
            cur_cls = detection['cls']
            cur_conf = detection['conf']
            
            # Skip low-confidence detections for new tracks
            if cur_conf < self.tracking_confidence_threshold and not valid_track_ids:
                continue
            
            matched = False
            best_score = float('inf')
            best_match_id = None
            
            for track_id in valid_track_ids:
                if track_id in used_prev_ids:
                    continue
                
                prev_data = self.prev_detections[track_id]
                prev_pos = prev_data['position']
                prev_cls = prev_data.get('cls', cur_cls)
                
                # 1. Calculate spatial distance
                spatial_dist = np.sqrt((cur_pos[0] - prev_pos[0])**2 + (cur_pos[1] - prev_pos[1])**2)
                
                # 2. Adaptive distance threshold based on vehicle type and stability
                base_dist = 100
                if cur_cls in [2, 3]:  # cars, motorcycles - smaller, potentially faster
                    max_dist = base_dist * 1.2
                elif cur_cls in [5, 7]:  # buses, trucks - larger, slower
                    max_dist = base_dist * 0.9
                else:
                    max_dist = base_dist
                
                # Adjust threshold based on track stability
                stability = track_stability_scores.get(track_id, 0.5)
                adjusted_max_dist = max_dist * (0.7 + 0.6 * stability)  # More stable tracks get larger search radius
                
                if spatial_dist > adjusted_max_dist:
                    continue
                
                # 3. Class consistency bonus
                class_consistency = 0.8 if cur_cls == prev_cls else 1.2
                
                # 4. Direction consistency (if available)
                direction_consistency = 1.0
                if track_id in self.vehicle_directions:
                    historical_direction = self.get_direction_from_history(track_id)
                    if historical_direction:
                        # Predict where the vehicle should be based on previous direction
                        hist_dx, hist_dy = historical_direction
                        predicted_pos = (prev_pos[0] + hist_dx * 0.5, prev_pos[1] + hist_dy * 0.5)
                        predicted_dist = np.sqrt((cur_pos[0] - predicted_pos[0])**2 + (cur_pos[1] - predicted_pos[1])**2)
                        
                        # Reward predictions that are close to actual position
                        direction_consistency = 0.7 + 0.3 * np.exp(-predicted_dist / 50.0)
                
                # 5. Confidence consistency
                confidence_factor = min(cur_conf, 0.9) / 0.9  # Normalize confidence
                
                # 6. Track age factor (prefer older, more established tracks)
                track_age_factor = min(len(self.track_history.get(track_id, [])) / 5.0, 1.0)
                age_bonus = 0.9 + 0.1 * track_age_factor
                
                # Combine all factors into a comprehensive matching score
                matching_score = spatial_dist * class_consistency * direction_consistency / (confidence_factor * age_bonus * stability)
                
                if matching_score < best_score:
                    best_score = matching_score
                    best_match_id = track_id
            
            # Assign track ID based on best match
            if best_match_id is not None:
                track_id = best_match_id
                used_prev_ids.add(track_id)
                prev_pos = self.prev_detections[track_id]['position']
                
                # Enhanced direction calculation using track history
                historical_direction = self.get_direction_from_history(track_id)
                
                # Current frame movement
                dx = cur_pos[0] - prev_pos[0]
                dy = cur_pos[1] - prev_pos[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                
                if magnitude > 0.3 or historical_direction:  # Lower threshold for movement detection
                    if historical_direction and magnitude > 0.3:
                        # Blend current movement with historical direction
                        hist_dx, hist_dy = historical_direction
                        hist_magnitude = np.sqrt(hist_dx**2 + hist_dy**2)
                        
                        if hist_magnitude > 0.1:
                            # Weighted combination: 70% history, 30% current
                            blended_dx = 0.7 * hist_dx + 0.3 * dx
                            blended_dy = 0.7 * hist_dy + 0.3 * dy
                            blended_magnitude = np.sqrt(blended_dx**2 + blended_dy**2)
                            blended_angle = np.degrees(np.arctan2(blended_dy, blended_dx))
                        else:
                            blended_dx, blended_dy = dx, dy
                            blended_magnitude = magnitude
                            blended_angle = np.degrees(np.arctan2(dy, dx))
                    else:
                        blended_dx, blended_dy = dx, dy
                        blended_magnitude = magnitude
                        blended_angle = np.degrees(np.arctan2(dy, dx)) if magnitude > 0.3 else 0
                    
                    # Temporal smoothing with previous direction
                    if track_id in self.vehicle_directions and 'smoothed_vector' in self.vehicle_directions[track_id]:
                        prev_direction = self.vehicle_directions[track_id]
                        prev_dx, prev_dy = prev_direction['smoothed_vector']
                        
                        # Adaptive smoothing based on magnitude change
                        magnitude_change = abs(blended_magnitude - prev_direction.get('smoothed_magnitude', blended_magnitude))
                        smoothing_factor = max(0.3, min(0.8, 0.5 + magnitude_change / 20.0))
                        
                        smoothed_dx = smoothing_factor * blended_dx + (1 - smoothing_factor) * prev_dx
                        smoothed_dy = smoothing_factor * blended_dy + (1 - smoothing_factor) * prev_dy
                        smoothed_magnitude = np.sqrt(smoothed_dx**2 + smoothed_dy**2)
                        smoothed_angle = np.degrees(np.arctan2(smoothed_dy, smoothed_dx))
                    else:
                        smoothed_dx, smoothed_dy = blended_dx, blended_dy
                        smoothed_magnitude = blended_magnitude
                        smoothed_angle = blended_angle
                    
                    self.vehicle_directions[track_id] = {
                        'vector': (dx, dy),
                        'blended_vector': (blended_dx, blended_dy),
                        'smoothed_vector': (smoothed_dx, smoothed_dy),
                        'magnitude': magnitude,
                        'blended_magnitude': blended_magnitude,
                        'smoothed_magnitude': smoothed_magnitude,
                        'angle': np.degrees(np.arctan2(dy, dx)) if magnitude > 0.3 else 0,
                        'blended_angle': blended_angle,
                        'smoothed_angle': smoothed_angle
                    }
                
                # Update track history with new position
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(cur_pos)
                if len(self.track_history[track_id]) > self.max_track_history:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
            else:
                # Create new track only for high-confidence detections
                if cur_conf >= self.tracking_confidence_threshold:
                    track_id = next_track_id
                    next_track_id += 1
                else:
                    # Skip low-confidence detections that don't match existing tracks
                    continue
            
            detection['track_id'] = track_id
            self.prev_detections[track_id] = {
                'position': cur_pos,
                'time': current_time,
                'cls': cur_cls,
                'conf': cur_conf
            }
            matched_detections.append(detection)

        # Enhanced cleanup: Remove tracks with consistently low confidence or poor stability
        tracks_to_remove = []
        for track_id in list(self.prev_detections.keys()):
            if track_id not in [det['track_id'] for det in matched_detections]:
                stability = track_stability_scores.get(track_id, 0.0)
                if stability < 0.3:  # Remove unstable tracks
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.prev_detections.pop(track_id, None)
            self.vehicle_directions.pop(track_id, None)
            self.track_history.pop(track_id, None)

        return matched_detections

    def detect_vehicles_with_bytetrack(self, frame, frame_timestamp=None):
        """
        Enhanced vehicle detection using ByteTrack with VehicleDetectionTracker features.
        Args:
            frame: Input image frame from CARLA
            frame_timestamp: Timestamp for speed calculation (optional)
        Returns:
            dict: Comprehensive detection results including vehicle details, speeds, and classifications
        """
        if self.yolo_model is None:
            return {
                "error": "YOLO model not available",
                "number_of_vehicles_detected": 0,
                "detected_vehicles": [],
                "annotated_frame_base64": None,
                "original_frame_base64": self._encode_image_base64(frame)
            }
        
        if frame_timestamp is None:
            frame_timestamp = datetime.now()
        
        # Initialize response structure
        response = {
            "number_of_vehicles_detected": 0,
            "detected_vehicles": [],
            "annotated_frame_base64": None,
            "original_frame_base64": None
        }
        
        try:
            # Enhance brightness for better detection
            enhanced_frame = self._increase_brightness(frame)
            
            # Run YOLO with ByteTrack tracking
            if self.use_bytetrack:
                results = self.yolo_model.track(
                    enhanced_frame, 
                    persist=True, 
                    tracker=self.tracker_config,
                    conf=0.12,
                    classes=[1,2,3,5,7,9,10],  # Vehicle classes
                    iou=0.45,
                    max_det=25,
                    verbose=False
                )
            else:
                results = self.yolo_model(
                    enhanced_frame,
                    conf=0.12,
                    classes=[1,2,3,5,7,9,10],
                    iou=0.45,
                    max_det=25,
                    verbose=False
                )
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if tracking was successful (ByteTrack)
                if (hasattr(result, 'boxes') and result.boxes is not None and 
                    hasattr(result.boxes, 'id') and result.boxes.id is not None):
                    
                    # Extract detection data
                    boxes = result.boxes.xywh.cpu()
                    conf_list = result.boxes.conf.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    clss = result.boxes.cls.cpu().tolist()
                    names = result.names
                    
                    # Get annotated frame with tracking lines
                    annotated_frame = result.plot()
                    
                    # Process each detection
                    for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
                        x, y, w, h = box
                        label = str(names[cls])
                        
                        # Update track history for polyline drawing
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        
                        track = self.track_history[track_id]
                        track.append((float(x), float(y)))
                        
                        # Limit track history
                        if len(track) > self.max_track_history_bytetrack:
                            track.pop(0)
                        
                        # Draw tracking polyline
                        if len(track) > 1:
                            from ultralytics.utils.plotting import colors
                            bbox_color = colors(cls, True)
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=False, 
                                        color=bbox_color, thickness=self.track_thickness)
                        
                        # Calculate speed and direction
                        speed_info = self._calculate_speed_and_direction(
                            track_id, (float(x), float(y)), frame_timestamp
                        )
                        
                        # Extract vehicle frame for classification
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        
                        # Ensure coordinates are within frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        vehicle_frame = frame[y1:y2, x1:x2]
                        vehicle_frame_base64 = self._encode_image_base64(vehicle_frame)
                        
                        # Classify vehicle (color and model)
                        color_info_json, model_info_json = self._classify_vehicle(vehicle_frame, track_id)
                        
                        # Add vehicle to detected set
                        self.detected_vehicles.add(track_id)
                        response["number_of_vehicles_detected"] += 1
                        
                        # Build vehicle information
                        vehicle_info = {
                            "vehicle_id": track_id,
                            "vehicle_type": label,
                            "detection_confidence": conf.item(),
                            "vehicle_coordinates": {
                                "x": x.item(),
                                "y": y.item(),
                                "width": w.item(),
                                "height": h.item()
                            },
                            "vehicle_frame_base64": vehicle_frame_base64,
                            "vehicle_frame_timestamp": frame_timestamp,
                            "speed_info": speed_info
                        }
                        
                        # Add classification results if available
                        if color_info_json:
                            vehicle_info["color_info"] = color_info_json
                        if model_info_json:
                            vehicle_info["model_info"] = model_info_json
                        
                        response["detected_vehicles"].append(vehicle_info)
                    
                    # Encode annotated frame
                    response["annotated_frame_base64"] = self._encode_image_base64(annotated_frame)
                
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # Fallback to regular detection without tracking
                    response["annotated_frame_base64"] = self._encode_image_base64(result.plot())
                    response["number_of_vehicles_detected"] = len(result.boxes)
            
            # Encode original frame
            response["original_frame_base64"] = self._encode_image_base64(frame)
            
        except Exception as e:
            print(f"⚠️ Vehicle detection with ByteTrack failed: {e}")
            response["error"] = str(e)
            response["original_frame_base64"] = self._encode_image_base64(frame)
        
        return response

    def detect_vehicles(self, frame):
        """
        Enhanced vehicle detection with adaptive frame skipping, memory optimization, and improved tracking.
        Args:
            frame: Input image frame from CARLA
        Returns:
            Tuple of (processed frame with detections, vehicle count)
        """
        if self.yolo_model is None:
            print("❌ YOLO model is None - detection disabled")
            return frame, 0
        
        # Quick test to make sure YOLO model is functional
        try:
            # Test if model is actually loaded and functional
            if not hasattr(self.yolo_model, 'predict') and not callable(self.yolo_model):
                print("❌ YOLO model is not functional")
                return frame, 0
        except Exception as e:
            print(f"❌ YOLO model test failed: {e}")
            return frame, 0
        
        start_time = time.time()
        
        # Adaptive frame skipping based on processing performance
        if self.adaptive_frame_skip:
            # Calculate current frame skip rate based on recent performance
            if len(self.processing_times) > 5:
                avg_time = sum(self.processing_times[-5:]) / 5
                if avg_time > self.max_processing_time:
                    self.frame_skip_rate = min(self.base_frame_skip_rate + 1, 4)
                else:
                    self.frame_skip_rate = max(self.base_frame_skip_rate, 1)
        
        # Implement adaptive frame skipping
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.frame_skip_rate:
            if self.last_processed_frame is not None:
                return self.last_processed_frame, len(self.prev_detections)
            else:
                return frame, 0
        
        self.frame_skip_counter = 0  # Reset counter
        
        # Reset detection statistics
        self.detection_stats = {
            "total_vehicles": 0,
            "close_vehicles": 0,
            "medium_vehicles": 0,
            "far_vehicles": 0,
            "processing_time": 0.0
        }
        
        # Get image dimensions
        height, width = frame.shape[:2]
        image_center = (width // 2, height // 2)
        
        # Smart resolution scaling based on frame complexity
        target_w, target_h = 640, 360  # Base resolution for better detection
        
        # Apply preprocessing for better stationary vehicle detection
        enhanced_frame = self.preprocess_image_for_stationary(frame)
        small_frame = cv2.resize(enhanced_frame, (target_w, target_h))
        scale_x = width / target_w
        scale_y = height / target_h
        
        try:
            # Memory optimization: Clear GPU cache before detection
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"🔍 Running YOLO detection on frame {small_frame.shape}...")  # Debug
            
            # Run detection with optimized settings
            results = self.yolo_model(
                small_frame,
                conf=0.12,           # Lower confidence for better recall
                classes=[1,2,3,5,7,9,10],  # Vehicle classes (bicycle, car, motorcycle, bus, truck, boat, traffic_light)
                iou=0.45,            # IoU threshold for NMS
                max_det=25,          # Increased max detections for better coverage
                verbose=False,
                device='cpu'         # Force CPU inference
            )
            
            print(f"✅ YOLO completed. Found {len(results)} result objects")  # Debug
            
            # Memory optimization: Release tensors immediately
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if hasattr(result.boxes, 'data'):
                        result.boxes.data = result.boxes.data.cpu().detach()
                if hasattr(result, 'obb') and result.obb is not None:
                    if hasattr(result.obb, 'data'):
                        result.obb.data = result.obb.data.cpu().detach()
                        
        except Exception as e:
            print(f'⚠️ Skipped frame due to detection error: {type(e).__name__}: {str(e)[:100]}')
            # Track processing time even for failed frames
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times = self.processing_times[-10:]
            return frame, 0
        # Create a copy of the frame to draw on
        output_frame = frame.copy()
        # Store current detections for tracking
        current_detections = []
        # Process results
        if len(results) > 0:
            result = results[0]
            # Standard boxes processing (fixed)
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                print(f"🔍 Found {len(result.boxes)} detections")  # Debug
                
                # Get all detection data at once
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                conf_scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls) in enumerate(zip(boxes_xyxy, conf_scores, class_ids)):
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Calculate center
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    current_detections.append({
                        'position': (cx, cy),
                        'box': (x1, y1, x2, y2),
                        'conf': float(conf),
                        'cls': int(cls)
                    })
                    
                print(f"✅ Processed {len(current_detections)} valid detections")  # Debug
            else:
                print("⚠️ No detection boxes found in YOLO results")  # Debug
        # Match detections with previous frames to get tracking IDs and directions
        if current_detections:
            print(f"🎯 Matching {len(current_detections)} detections...")  # Debug
            matched_detections = self._match_detections(current_detections)
            print(f"✅ Matched {len(matched_detections)} detections with tracking IDs")  # Debug
            
            for detection in matched_detections:
                cx, cy = detection['position']
                track_id = detection['track_id']
                conf = detection['conf']
                cls = detection['cls']
                box_coords = detection['box']
                
                # Vehicle type filtering and color coding
                vehicle_type = self.vehicle_type_names.get(cls, f"class_{cls}")
                if cls == 2:  # car
                    type_color = (0, 255, 255)  # Yellow
                elif cls in [5, 7]:  # bus, truck
                    type_color = (255, 0, 255)  # Magenta
                elif cls in [1, 3]:  # bicycle, motorcycle
                    type_color = (0, 255, 0)   # Green
                else:
                    type_color = (128, 128, 128)  # Gray
                
                # Distance-based classification
                distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
                if distance < self.close_threshold:
                    color = (0, 0, 255)  # Red
                    distance_label = "CLOSE"
                    self.detection_stats["close_vehicles"] += 1
                elif distance < self.medium_threshold:
                    color = (0, 165, 255)  # Orange
                    distance_label = "MEDIUM"
                    self.detection_stats["medium_vehicles"] += 1
                else:
                    color = (0, 255, 0)  # Green
                    distance_label = "FAR"
                    self.detection_stats["far_vehicles"] += 1
                
                # Draw bounding box with type-specific color
                if box_coords:
                    x1, y1, x2, y2 = box_coords
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    # Add vehicle type indicator
                    cv2.rectangle(output_frame, (x1, y1-20), (x1+80, y1), type_color, -1)
                    cv2.putText(output_frame, vehicle_type, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw center point
                cv2.circle(output_frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Enhanced direction visualization with improved smoothing
                direction_text = "N/A"
                if track_id in self.vehicle_directions:
                    direction = self.vehicle_directions[track_id]
                    # Use blended direction for better accuracy, smoothed for visualization
                    if 'blended_magnitude' in direction and direction['blended_magnitude'] > 0.3:
                        # Use smoothed direction for arrow visualization
                        if 'smoothed_vector' in direction:
                            smoothed_dx, smoothed_dy = direction['smoothed_vector']
                            smoothed_magnitude = direction['smoothed_magnitude']
                            
                            if smoothed_magnitude > 0.1:
                                scale = 45.0  # Larger arrows for better visibility
                                arrow_dx = int(smoothed_dx / smoothed_magnitude * scale)
                                arrow_dy = int(smoothed_dy / smoothed_magnitude * scale)
                                
                                # Draw smoothed direction arrow with type-specific color
                                cv2.arrowedLine(output_frame, (cx, cy), (cx + arrow_dx, cy + arrow_dy), 
                                              type_color, 3, tipLength=0.3)
                        
                        # Use blended angle for text (more accurate than smoothed for direction classification)
                        angle = direction.get('blended_angle', direction.get('smoothed_angle', 0))
                        if -22.5 <= angle < 22.5:
                            direction_text = "→"
                        elif 22.5 <= angle < 67.5:
                            direction_text = "↘"
                        elif 67.5 <= angle < 112.5:
                            direction_text = "↓"
                        elif 112.5 <= angle < 157.5:
                            direction_text = "↙"
                        elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                            direction_text = "←"
                        elif -157.5 <= angle < -112.5:
                            direction_text = "↖"
                        elif -112.5 <= angle < -67.5:
                            direction_text = "↑"
                        elif -67.5 <= angle < -22.5:
                            direction_text = "↗"
                    elif 'magnitude' in direction and direction['magnitude'] > 0.1:
                        # Fallback to basic direction if advanced data not available
                        direction_text = "⚬"  # Slight movement indicator
                
                # Enhanced track history visualization
                if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], np.int32)
                    points = points.reshape((-1, 1, 2))
                    # Color-coded track history based on vehicle type
                    cv2.polylines(output_frame, [points], False, type_color, 2)
                
                # Enhanced label with vehicle type and smoothed direction
                label = f"ID:{track_id} {vehicle_type} - {distance_label} - Dir:{direction_text} ({conf:.2f})"
                cv2.putText(output_frame, label, (cx, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                self.detection_stats["total_vehicles"] += 1
        
        # Draw distance threshold circles
        cv2.circle(output_frame, image_center, self.close_threshold, (0, 0, 255), 1)
        cv2.circle(output_frame, image_center, self.medium_threshold, (0, 165, 255), 1)
        
        # Enhanced summary with performance info
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times = self.processing_times[-10:]
        
        avg_fps = 1.0 / (sum(self.processing_times) / len(self.processing_times)) if self.processing_times else 0
        summary = f"Vehicles: {self.detection_stats['total_vehicles']} " + \
                 f"(CLOSE: {self.detection_stats['close_vehicles']}, " + \
                 f"MEDIUM: {self.detection_stats['medium_vehicles']}, " + \
                 f"FAR: {self.detection_stats['far_vehicles']}) | " + \
                 f"FPS: {avg_fps:.1f} | Skip: {self.frame_skip_rate}"
        cv2.putText(output_frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance indicator
        performance_color = (0, 255, 0) if processing_time < self.max_processing_time else (0, 0, 255)
        cv2.circle(output_frame, (width - 30, 30), 8, performance_color, -1)
        
        self.detection_stats["processing_time"] = processing_time
        
        # Memory cleanup
        del results, small_frame, enhanced_frame
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Store the processed frame for frame skipping
        self.last_processed_frame = output_frame.copy()
        
        return output_frame, self.detection_stats["total_vehicles"]
    
    def process_top_view(self, image, ego_velocity=None, ego_angular=None):
        """Process top view camera image with vehicle detection."""
        if image is None:
            return None
        # Vehicle detection with YOLOv11m-obb (ego_velocity and ego_angular ignored for now)
        processed_image, vehicle_count = self.detect_vehicles(image)
        return processed_image
        
    def process_front_view(self, image):
        """Process front view camera image (simple pass-through)."""
        # For front view, we're just returning the image without processing
        return image
        
    def process_side_view(self, image, side):
        """Process side view camera image (simple pass-through)."""
        # For side views, we're just returning the image without processing
        return image
        
    def process_rear_view(self, image):
        """Process rear view camera image (simple pass-through)."""
        # For rear view, we're just returning the image without processing
        return image
    
    def process_video(self, video_path, result_callback):
        """
        Process a video file frame by frame using ByteTrack features.
        Args:
            video_path (str): Path to the video file
            result_callback (function): Callback function to handle results for each frame
        """
        print(f"🎥 Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames at {fps:.2f} FPS")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                timestamp = datetime.now()
                
                # Process frame with ByteTrack
                response = self.detect_vehicles_with_bytetrack(frame, timestamp)
                
                # Display annotated frame if available
                if 'annotated_frame_base64' in response and response['annotated_frame_base64']:
                    annotated_frame = self._decode_image_base64(response['annotated_frame_base64'])
                    if annotated_frame is not None:
                        # Add frame info overlay
                        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Vehicles: {response['number_of_vehicles_detected']}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Video Detection Tracker - Enhanced CARLA", annotated_frame)
                
                # Call the callback with the response
                if result_callback:
                    result_callback(response)
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("🛑 Video processing stopped by user")
                    break
                    
        except KeyboardInterrupt:
            print("🛑 Video processing interrupted by user")
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Video processing completed. Processed {frame_count} frames.")
    
    def process_frame_base64(self, frame_base64, frame_timestamp=None):
        """
        Process a base64-encoded frame for web/API integration.
        Args:
            frame_base64 (str): Base64-encoded input frame
            frame_timestamp: Optional timestamp for the frame
        Returns:
            dict: Processing results or error message
        """
        frame = self._decode_image_base64(frame_base64)
        if frame is not None:
            return self.detect_vehicles_with_bytetrack(frame, frame_timestamp)
        else:
            return {"error": "Failed to decode the base64 image"}
    
    def get_vehicle_analytics(self):
        """
        Get comprehensive analytics about detected vehicles.
        Returns:
            dict: Analytics including counts, speeds, directions, etc.
        """
        analytics = {
            "total_unique_vehicles": len(self.detected_vehicles),
            "active_tracks": len(self.track_history),
            "vehicles_with_speed_data": len(self.vehicle_speeds),
            "vehicles_with_color_data": len(self.vehicle_color_info),
            "vehicles_with_model_data": len(self.vehicle_model_info),
            "detection_stats": self.detection_stats.copy()
        }
        
        # Add speed statistics
        if self.vehicle_speeds:
            speeds = [info.get("kph", 0) for info in self.vehicle_speeds.values() if info.get("kph") is not None]
            if speeds:
                analytics["speed_stats"] = {
                    "avg_speed_kph": sum(speeds) / len(speeds),
                    "max_speed_kph": max(speeds),
                    "min_speed_kph": min(speeds)
                }
        
        return analytics
    
    def cleanup_old_tracks(self, max_age_seconds=30):
        """
        Clean up old tracking data to prevent memory leaks.
        Args:
            max_age_seconds (int): Maximum age for tracks in seconds
        """
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id in list(self.vehicle_timestamps.keys()):
            timestamps = self.vehicle_timestamps[track_id]["timestamps"]
            if timestamps:
                last_timestamp = timestamps[-1]
                if hasattr(last_timestamp, 'timestamp'):
                    age = current_time - last_timestamp.timestamp()
                else:
                    age = current_time - float(last_timestamp)
                
                if age > max_age_seconds:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.vehicle_timestamps.pop(track_id, None)
            self.vehicle_speeds.pop(track_id, None)
            self.vehicle_color_info.pop(track_id, None)
            self.vehicle_model_info.pop(track_id, None)
            self.track_history.pop(track_id, None)
            self.detected_vehicles.discard(track_id)
    
    def get_detection_stats(self):
        """Get the current detection statistics."""
        return self.detection_stats


def create_cv_processor():
    """Factory function to create a computer vision processor."""
    return ComputerVisionProcessor()


def test_cv_processor():
    """Test function for the computer vision processor."""
    print("🧪 Testing Computer Vision Processor...")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[200:280, 200:440] = [100, 100, 100]  # Simulate road
    test_image[240:250, 220:420] = [255, 255, 255]  # Simulate lane marking
    test_image[150:200, 300:350] = [0, 0, 255]      # Simulate red car
    
    processor = create_cv_processor()
    
    # Test processing
    result = processor.process_top_view(test_image)
    
    if result is not None:
        print("✅ Computer Vision Processor test passed!")
        stats = processor.get_detection_stats()
        print(f"   📊 Stats: {stats}")
        
        # Test YOLO availability
        if processor.yolo_model:
            print("   🚀 YOLO detection: READY")
    else:
        print("❌ Computer Vision Processor test failed!")


if __name__ == "__main__":
    test_cv_processor()
