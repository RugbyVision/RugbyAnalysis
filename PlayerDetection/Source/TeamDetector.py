import cv2 as cv
import numpy as np
import ColorChecking as col
import threading as th
import torch
import torchvision
from torchvision import transforms as t

"""def gamma_correction(img: cv.Mat, gamma:float = 1.):
    invGamma = 1. / gamma
    table = np.array([((i / 255.) ** invGamma) * 255. for i in np.arange(256)]).astype(np.uint8)
    return cv.LUT(img, table)

def get_color(i, cropped):
    thread_results[i] = col.detect_color(cropped, field_color)[0]

def detect_colors(img, data, ratio):
    global thread_results
    objects = {}
    boxes = data.boxes
    categories = data.cls

    threads = []
    for i, (label, box) in enumerate(zip(categories, boxes)):
        if label == "person":
            x0, y0 = box[0]
            x1, y1 = box[1]
            cropped = img[y0:y1, x0:x1]
            ch, cw, _ = cropped.shape
            cropped = cv.resize(cropped, (int(cw / ratio), int(ch / ratio)))
            cv.imshow("Y1", cropped)
            threads.append(th.Thread(target=get_color, args=(i, cropped)))
            objects[i] = {"object": "Person", "location": [x0, y0, x1, y1]}
    
    thread_results = [(0, 0, 0)] * (max(objects.keys()) + 1)
    [t.start() for t in threads]
    print(f"{len(threads)} threads started", sep='\n')
    [t.join() for t in threads]
    print(f"{len(threads)} threads finished", sep='\n')

    for i in objects.keys():
        objects[i]["color"] = thread_results[i]

    all_colors = [color["color"] for color in objects.values()]

    groups = col.get_ranged_groups(all_colors, groups_color_filters)

    for key, value in objects.items():
        for k, v in groups.items():
            if value["color"] in v:
                objects[key]["team"] = k

    return img, objects"""


if __name__ == "__main__":
    path = "PlayerDetection/Res/Videos/RugbyGameSample_Aug_Cam_1_1.mp4"
    cap = cv.VideoCapture(path)

    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            raise

        adj = gamma_correction(frame, 2.)
        resized = cv.resize(adj, (1920, 1080), interpolation=cv.INTER_AREA)
        cv.imwrite("PlayerDetection/Res/Images/RugbyGameSample_Aug_Cam_1_1.jpg", resized)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")

    model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    model.to(device)
    model.eval()

    COCO_CategoryNames = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']