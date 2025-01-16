import cv2
import numpy as np
from pathlib import Path

def load_yolo():
    """Load YOLO model and configuration"""
    # Load COCO class labels
    labels_path = "coco.names"
    labels = open(labels_path).read().strip().split("\n")

    # Load YOLO configurations and weights
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    
    # Load YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, labels, output_layers

def detect_objects(frame, net, output_layers, labels, confidence_threshold=0.5):
    """Detect objects in the frame"""
    height, width = frame.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Forward pass through network
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Scale bounding box coordinates back to size of image
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Get top-left corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'label': labels[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for detection in detections:
        x, y, w, h = detection['box']
        label = f"{detection['label']}: {detection['confidence']:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Load YOLO model
    print("Loading YOLO model...")
    net, labels, output_layers = load_yolo()
    print("Model loaded successfully!")
    
    while True:
        # Read camera details
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects
        detections = detect_objects(frame, net, output_layers, labels)
        
        # Draw detections
        frame = draw_detections(frame, detections)
        
        # Show frame
        cv2.imshow("Object Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()