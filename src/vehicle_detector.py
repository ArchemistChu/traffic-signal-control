#!/usr/bin/env python3
"""
Vehicle Detection Simulation Module (VehicleDetector)
Uses ground-truth vehicle positions from SUMO to simulate YOLO detection effects
Adds detection noise and delay to simulate real-world perception uncertainty
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random

@dataclass
class DetectedVehicle:
    """Detected vehicle information"""
    vehicle_id: str
    position: Tuple[float, float]  # (x, y) coordinates
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) bounding box
    confidence: float  # Detection confidence [0, 1]
    vehicle_type: str  # Vehicle type
    speed: float  # Speed
    lane_id: str  # Lane ID
    timestamp: float  # Detection timestamp
    detection_delay: float  # Detection delay time

@dataclass
class DetectionConfig:
    """Detection configuration parameters"""
    # YOLO detection parameters
    confidence_threshold: float = 0.5  # Confidence threshold
    nms_threshold: float = 0.4  # Non-maximum suppression threshold
    detection_range: float = 200.0  # Detection range (meters)
    
    # Noise parameters
    position_noise_std: float = 0.5  # Position noise standard deviation (meters)
    bbox_noise_std: float = 0.3  # Bounding box noise standard deviation (meters)
    confidence_noise_std: float = 0.1  # Confidence noise standard deviation
    
    # Detection performance parameters
    detection_rate: float = 0.95  # Detection success rate (avoid missed detections)
    false_positive_rate: float = 0.02  # False positive rate
    
    # Delay parameters
    min_detection_delay: float = 0.05  # Minimum detection delay (seconds)
    max_detection_delay: float = 0.2   # Maximum detection delay (seconds)
    network_delay_std: float = 0.03    # Network transmission delay standard deviation
    
    # Vehicle dimension configuration (for generating bounding boxes)
    vehicle_dimensions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'passenger': (4.5, 1.8),  # length, width
        'truck': (12.0, 2.5),
        'bus': (18.0, 2.5),
        'motorcycle': (2.0, 0.8)
    })

class VehicleDetector:
    """
    Vehicle detection simulator
    Simulates YOLO-based vehicle detection system, including noise and delay
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize detector
        
        Args:
            config: detection configuration parameters
        """
        self.config = config or DetectionConfig()
        self.detection_history = deque(maxlen=1000)  # Detection history
        self.delayed_detections = deque()  # Delayed detection queue
        self.current_time = 0.0
        
        # Statistics
        self.total_detections = 0
        self.successful_detections = 0
        self.false_positives = 0
        self.missed_detections = 0
    
    def detect_vehicles(self, ground_truth_vehicles: List[Dict], 
                       current_time: float, 
                       camera_positions: List[Tuple[float, float]] = None) -> List[DetectedVehicle]:
        """
        Main detection function: convert ground-truth vehicle data to simulated detection results
        
        Args:
            ground_truth_vehicles: ground-truth vehicle data provided by SUMO
            current_time: current simulation time
            camera_positions: camera position list, uses default intersection position if None
            
        Returns:
            List[DetectedVehicle]: list of detected vehicles
        """
        self.current_time = current_time
        
        # Default intersection camera position
        if camera_positions is None:
            camera_positions = [(0.0, 0.0)]  # Intersection center
        
        all_detections = []
        
        # Process detection for each camera
        for camera_pos in camera_positions:
            camera_detections = self._process_camera_detection(
                ground_truth_vehicles, camera_pos
            )
            all_detections.extend(camera_detections)
        
        # Add false positive vehicles
        false_positives = self._generate_false_positives(camera_positions)
        all_detections.extend(false_positives)
        
        # Apply non-maximum suppression
        final_detections = self._apply_nms(all_detections)
        
        # Add detection delay
        delayed_detections = self._apply_detection_delay(final_detections)
        
        # Process delayed queue, return detection results that should be output at current time
        current_detections = self._process_delayed_detections()
        
        # Update statistics
        self._update_statistics(len(ground_truth_vehicles), len(current_detections))
        
        return current_detections
    
    def _process_camera_detection(self, ground_truth_vehicles: List[Dict], 
                                 camera_pos: Tuple[float, float]) -> List[DetectedVehicle]:
        """
        Process vehicle detection for a single camera
        
        Args:
            ground_truth_vehicles: ground-truth vehicle data
            camera_pos: camera position
            
        Returns:
            List[DetectedVehicle]: vehicles detected by this camera
        """
        detections = []
        
        for vehicle in ground_truth_vehicles:
            try:
                # Get vehicle ground-truth information
                veh_pos = vehicle.get('position', (0, 0))
                if isinstance(veh_pos, (tuple, list)) and len(veh_pos) >= 2:
                    x, y = float(veh_pos[0]), float(veh_pos[1])
                else:
                    continue  # Skip vehicles with invalid positions
                
                # Check if vehicle is within detection range
                distance = np.sqrt((x - camera_pos[0])**2 + (y - camera_pos[1])**2)
                if distance > self.config.detection_range:
                    continue
                
                # Simulate detection success rate (avoid missed detections)
                if random.random() > self.config.detection_rate:
                    self.missed_detections += 1
                    continue
                
                # Generate detection result
                detected_vehicle = self._create_detected_vehicle(vehicle, camera_pos)
                if detected_vehicle:
                    detections.append(detected_vehicle)
                    
            except Exception as e:
                print(f"Error processing vehicle detection: {e}")
                continue
        
        return detections
    
    def _create_detected_vehicle(self, ground_truth: Dict, 
                               camera_pos: Tuple[float, float]) -> Optional[DetectedVehicle]:
        """
        Create detection result from ground-truth vehicle data
        
        Args:
            ground_truth: ground-truth vehicle data
            camera_pos: camera position
            
        Returns:
            DetectedVehicle: detection result
        """
        try:
            # Extract ground-truth data
            vehicle_id = ground_truth.get('vehicle_id', 'unknown')
            true_pos = ground_truth.get('position', (0, 0))
            vehicle_type = ground_truth.get('type_id', 'passenger')
            speed = ground_truth.get('speed', 0.0)
            lane_id = ground_truth.get('lane_id', 'unknown')
            
            # Add position noise
            noisy_x = true_pos[0] + np.random.normal(0, self.config.position_noise_std)
            noisy_y = true_pos[1] + np.random.normal(0, self.config.position_noise_std)
            noisy_position = (noisy_x, noisy_y)
            
            # Generate bounding box
            bbox = self._generate_bbox(noisy_position, vehicle_type)
            
            # Generate confidence based on distance and vehicle size
            distance = np.sqrt((true_pos[0] - camera_pos[0])**2 + 
                             (true_pos[1] - camera_pos[1])**2)
            base_confidence = max(0.1, 1.0 - distance / self.config.detection_range)
            
            # Add confidence noise
            confidence = base_confidence + np.random.normal(0, self.config.confidence_noise_std)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Check if confidence meets threshold
            if confidence < self.config.confidence_threshold:
                return None
            
            # Generate detection delay
            detection_delay = np.random.uniform(
                self.config.min_detection_delay, 
                self.config.max_detection_delay
            ) + np.abs(np.random.normal(0, self.config.network_delay_std))
            
            return DetectedVehicle(
                vehicle_id=vehicle_id,
                position=noisy_position,
                bbox=bbox,
                confidence=confidence,
                vehicle_type=vehicle_type,
                speed=speed,
                lane_id=lane_id,
                timestamp=self.current_time,
                detection_delay=detection_delay
            )
            
        except Exception as e:
            print(f"Error creating detection result: {e}")
            return None
    
    def _generate_bbox(self, position: Tuple[float, float], 
                      vehicle_type: str) -> Tuple[float, float, float, float]:
        """
        Generate vehicle bounding box
        
        Args:
            position: vehicle center position
            vehicle_type: vehicle type
            
        Returns:
            Tuple: bounding box (x1, y1, x2, y2)
        """
        # Get vehicle dimensions
        dimensions = self.config.vehicle_dimensions.get(vehicle_type, (4.5, 1.8))
        length, width = dimensions
        
        # Add bounding box noise
        noisy_length = length + np.random.normal(0, self.config.bbox_noise_std)
        noisy_width = width + np.random.normal(0, self.config.bbox_noise_std)
        
        # Ensure dimensions are positive
        noisy_length = max(1.0, noisy_length)
        noisy_width = max(0.5, noisy_width)
        
        # Generate bounding box (assuming vehicle orientation is horizontal)
        x, y = position
        x1 = x - noisy_length / 2
        y1 = y - noisy_width / 2
        x2 = x + noisy_length / 2
        y2 = y + noisy_width / 2
        
        return (x1, y1, x2, y2)
    
    def _generate_false_positives(self, camera_positions: List[Tuple[float, float]]) -> List[DetectedVehicle]:
        """
        Generate false positive vehicles
        
        Args:
            camera_positions: camera position list
            
        Returns:
            List[DetectedVehicle]: false positive vehicle list
        """
        false_positives = []
        
        for camera_pos in camera_positions:
            # Decide whether to generate false positive based on false positive rate
            if random.random() < self.config.false_positive_rate:
                # Randomly generate position within detection range
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(20, self.config.detection_range)
                
                fp_x = camera_pos[0] + distance * np.cos(angle)
                fp_y = camera_pos[1] + distance * np.sin(angle)
                
                # Randomly select vehicle type
                vehicle_types = list(self.config.vehicle_dimensions.keys())
                fp_type = random.choice(vehicle_types)
                
                # Generate low-confidence false positive
                fp_confidence = random.uniform(0.5, 0.8)
                
                false_positive = DetectedVehicle(
                    vehicle_id=f"fp_{self.current_time}_{random.randint(1000, 9999)}",
                    position=(fp_x, fp_y),
                    bbox=self._generate_bbox((fp_x, fp_y), fp_type),
                    confidence=fp_confidence,
                    vehicle_type=fp_type,
                    speed=0.0,
                    lane_id="unknown",
                    timestamp=self.current_time,
                    detection_delay=np.random.uniform(0.05, 0.15)
                )
                
                false_positives.append(false_positive)
                self.false_positives += 1
        
        return false_positives
    
    def _apply_nms(self, detections: List[DetectedVehicle]) -> List[DetectedVehicle]:
        """
        Apply non-maximum suppression to remove duplicate detections
        
        Args:
            detections: original detection list
            
        Returns:
            List[DetectedVehicle]: detection list after NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        final_detections = []
        
        while sorted_detections:
            # Select detection with highest confidence
            best_detection = sorted_detections.pop(0)
            final_detections.append(best_detection)
            
            # Remove detections overlapping with best detection
            remaining_detections = []
            for detection in sorted_detections:
                iou = self._calculate_iou(best_detection.bbox, detection.bbox)
                if iou < self.config.nms_threshold:
                    remaining_detections.append(detection)
            
            sorted_detections = remaining_detections
        
        return final_detections
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate IoU (Intersection over Union) of two bounding boxes
        
        Args:
            bbox1, bbox2: bounding boxes (x1, y1, x2, y2)
            
        Returns:
            float: IoU value
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_detection_delay(self, detections: List[DetectedVehicle]):
        """
        Apply detection delay
        
        Args:
            detections: current detection results
        """
        for detection in detections:
            # Calculate time when detection result should be output
            output_time = self.current_time + detection.detection_delay
            self.delayed_detections.append((output_time, detection))
    
    def _process_delayed_detections(self) -> List[DetectedVehicle]:
        """
        Process delayed detection queue, return detection results that should be output at current time
        
        Returns:
            List[DetectedVehicle]: detection results for current time
        """
        current_detections = []
        
        # Extract detections that should be output at current time from delayed queue
        remaining_detections = deque()
        
        while self.delayed_detections:
            output_time, detection = self.delayed_detections.popleft()
            if output_time <= self.current_time:
                current_detections.append(detection)
            else:
                remaining_detections.append((output_time, detection))
        
        self.delayed_detections = remaining_detections
        return current_detections
    
    def _update_statistics(self, total_vehicles: int, detected_vehicles: int):
        """Update detection statistics"""
        self.total_detections += total_vehicles
        self.successful_detections += detected_vehicles
    
    def get_detection_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dict: statistics information
        """
        if self.total_detections == 0:
            return {}
        
        detection_rate = self.successful_detections / self.total_detections
        
        return {
            'total_vehicles': self.total_detections,
            'successful_detections': self.successful_detections,
            'missed_detections': self.missed_detections,
            'false_positives': self.false_positives,
            'detection_rate': detection_rate,
            'false_positive_rate': self.false_positives / max(1, self.successful_detections),
            'precision': self.successful_detections / max(1, self.successful_detections + self.false_positives)
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.total_detections = 0
        self.successful_detections = 0
        self.false_positives = 0
        self.missed_detections = 0
    
    def set_detection_config(self, config: DetectionConfig):
        """Update detection configuration"""
        self.config = config


def main():
    """Test vehicle detection simulator"""
    print("Vehicle Detection Simulator Test")
    
    # Create detector
    detector = VehicleDetector()
    
    # Simulate vehicle data
    test_vehicles = [
        {
            'vehicle_id': 'veh1',
            'position': (10.0, 5.0),
            'type_id': 'passenger',
            'speed': 12.5,
            'lane_id': 'east_in_0'
        },
        {
            'vehicle_id': 'veh2',
            'position': (-15.0, -8.0),
            'type_id': 'truck',
            'speed': 8.0,
            'lane_id': 'west_in_0'
        }
    ]
    
    # Run detection
    current_time = 10.0
    detections = detector.detect_vehicles(test_vehicles, current_time)
    
    print(f"\nDetection results (time: {current_time:.1f}s):")
    for detection in detections:
        print(f"- Vehicle ID: {detection.vehicle_id}")
        print(f"  Position: ({detection.position[0]:.2f}, {detection.position[1]:.2f})")
        print(f"  Confidence: {detection.confidence:.3f}")
        print(f"  Type: {detection.vehicle_type}")
        print(f"  Delay: {detection.detection_delay:.3f}s")
        print()
    
    # Output statistics
    stats = detector.get_detection_statistics()
    print("Detection statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")

if __name__ == "__main__":
    main()
