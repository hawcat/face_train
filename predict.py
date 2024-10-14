from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train17/weights/best.pt")  # load a custom model

# Predict with the model
results = model("test.jpg")  # predict on an image
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
