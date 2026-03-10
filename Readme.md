Final Year Project

## Generate Accuracy Graphs

This project runs YOLOv3 inference in `main.py`.  
To evaluate on a labeled dataset and generate graphs, use `evaluate.py`.

### Expected dataset format

- Images: any nested structure inside `images` directory
- Labels: YOLO `.txt` files inside `labels` directory with the same relative path/name as image
- Label row format: `class_id x_center y_center width height`

Example:

```text
dataset/
  images/
    test1.jpg
    room/0001.png
  labels/
    test1.txt
    room/0001.txt
```

### Run evaluation

```bash
python evaluate.py \
  --images-dir /path/to/dataset/images \
  --labels-dir /path/to/dataset/labels \
  --cfg yolov3.cfg \
  --weights yolov3.weights \
  --names coco.names \
  --output-dir evaluation_outputs
```

### Generated outputs

- `evaluation_outputs/precision_recall_curve.png`
- `evaluation_outputs/f1_vs_confidence.png`
- `evaluation_outputs/confusion_matrix.png`
- `evaluation_outputs/threshold_metrics.csv`
- `evaluation_outputs/summary.json`

