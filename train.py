from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO(
        "runs\\detect\\train17\\weights\\last.pt"
    )  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="datasets\YOLODataset\dataset.yaml", epochs=400, imgsz=640, resume=True
    )
