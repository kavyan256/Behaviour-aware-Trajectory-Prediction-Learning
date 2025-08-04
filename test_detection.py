#!/usr/bin/env python3
"""
Simple test script to verify vehicle detection is working
"""

import cv2
import numpy as np
from computer_vision import ComputerVisionProcessor

def test_detection():
    """Test vehicle detection on a synthetic image."""
    print("🧪 Testing Vehicle Detection...")
    
    # Create processor
    cv_processor = ComputerVisionProcessor()
    
    # Check if YOLO is available
    if cv_processor.yolo_model is None:
        print("❌ YOLO model not available")
        return False
    
    print("✅ YOLO model loaded")
    
    # Create a test image with some shapes that might look like vehicles
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some rectangular shapes that might trigger vehicle detection
    cv2.rectangle(test_image, (100, 100), (180, 140), (128, 128, 128), -1)  # Gray rectangle
    cv2.rectangle(test_image, (300, 200), (400, 260), (64, 64, 128), -1)    # Another rectangle
    cv2.rectangle(test_image, (450, 300), (550, 350), (96, 96, 96), -1)     # Third rectangle
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)
    
    print("🔍 Running detection on test image...")
    
    # Run detection
    processed_frame, vehicle_count = cv_processor.detect_vehicles(test_image)
    
    print(f"📊 Detection result: {vehicle_count} vehicles detected")
    
    # Display the result
    cv2.imshow("Test Detection", processed_frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()
    
    # Test with a real image if we have one
    try:
        # Try to load a sample image (if it exists)
        sample_path = "sample_carla_frame.jpg"
        if cv2.imread(sample_path) is not None:
            real_image = cv2.imread(sample_path)
            print(f"🖼️ Testing on real image: {sample_path}")
            processed_real, real_count = cv_processor.detect_vehicles(real_image)
            print(f"📊 Real image result: {real_count} vehicles detected")
            
            cv2.imshow("Real Detection", processed_real)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
    except:
        print("ℹ️ No sample image found, skipping real image test")
    
    return True

if __name__ == "__main__":
    test_detection()
