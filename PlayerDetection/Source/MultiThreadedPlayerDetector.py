# import video import/export libraries
import sys, os
import cv2 as cv
import numpy as np
from PIL import Image
import h5py

# import NN-related libraries
import torch
import torchvision
from torchvision import transforms as t

# import libraries related to video processing
from queue import Queue
from threading import Thread, Event
from imutils.video import FPS
import time

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")
model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(weights="COCO_V1")
model.to(device)
model.eval()

FullQueue = Event()
FullQueue.clear()

COCO_CategoryNames = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def GetPred(Img, Thr):
    img = cv.cvtColor(Img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transform = t.Compose([t.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [COCO_CategoryNames[i] for i in list(pred[0]["labels"].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]["boxes"].cpu().detach().numpy())]
    pred_score = list(pred[0]["scores"].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > Thr]
    pred_boxes = [pred_boxes[i] for i in pred_t]
    pred_class = [pred_class[i] for i in pred_t]
    return pred_boxes, pred_class

def ObjectDetectionAPI(img, Thr=0.3, RectTh=3, TxtSize=3, TxtTh=3):
    boxes, pred_cls = GetPred(img, Thr)

    coords = None

    for i in range(len(boxes)):
        if pred_cls[i] == "sports ball":
            coords = list(boxes[i][0])
            for j, c in enumerate(boxes[i][1]):
                coords[j] += c
                coords[j] /= 2.
    
    return img, coords

class FileVideoStream:
    def __init__(self, path, queueSize=2048):
        self.path = path
        self.stream = cv.VideoCapture(path)
        self.halted = False
        self.Q = Queue(maxsize=queueSize)
        self.lastFrameCoords = None
        self.DetectedArray = None

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        idx = 0
        with h5py.File(self.path.replace(".mp4", ".h5").replace("Videos", "Arrays"), 'w') as hf:
            while True:
                if self.halted:
                    return
            
                if not self.Q.full():
                    (grabbed, frame) = self.stream.read()

                    if not grabbed:
                        self.halt()
                        # self.saveArray(np.array(Detected), "res/Arrays/Detected_rcnn")
                        FullQueue.set()
                        return

                    """img, coords = self.detectBall(frame, lastCoords=self.lastFrameCoords)
                    if coords is None:
                        coords = [None, None]
                    else:
                        self.lastFrameCoords = coords
                        center = tuple(map(int, coords))
                        cv.circle(img, center, 10, (0, 255, 0), 3)
                    Detected.append(coords)"""
                    img, coordArray = self.detectPeople(frame)
                    self.Q.put(img)
                    hf["Frame_" + str(idx) + "_People"] = coordArray

                    print(f"read the frame {idx}")
                    idx += 1
                else:
                    FullQueue.set()

    def read(self):
        return self.Q.get()
    
    def more(self):
        return self.Q.qsize() > 0
    
    def halt(self):
        self.halted = True

    def saveArray(self, arr, path):
        np.save(path, arr)
        print(arr)

    def detectBall(self, img, thr=0.5, lastCoords=None):
        boxes, pred_cls = GetPred(img, thr)

        coords = None

        dist = sys.float_info.max
        for i in range(len(boxes)):
            if pred_cls[i] == "sports ball":
                cv.rectangle(img, tuple(map(int, boxes[i][0])), tuple(map(int, boxes[i][1])), (0, 255, 0))
                coords = list(boxes[i][0])
                for j, c in enumerate(boxes[i][1]):
                    coords[j] += c
                    coords[j] /= 2.
                break
            elif lastCoords is not None and pred_cls[i] == "person":
                placement = list(boxes[i][0])
                for j, c in enumerate(boxes[i][1]):
                    placement[j] += c
                    placement[j] /= 2.
                offsetV = [a - b for a, b in zip(placement, lastCoords)]
                d = np.linalg.norm(np.array(offsetV))
                if dist >= d:
                    coords = placement
                    dist = d
    
    def detectPeople(self, img, thr=0.5, rectTh=3, rectCol=(0, 255, 0), txtFont=cv.FONT_HERSHEY_SIMPLEX, txtSize=3, txtTh=3):
        boxes, pred_cls = GetPred(img, thr)
        coords = []

        for i in range(len(boxes)):
            if pred_cls[i] == "person":
                cv.rectangle(img, tuple(map(int, boxes[i][0])), tuple(map(int, boxes[i][1])), rectCol, rectTh)
                cv.putText(img, "person", tuple(map(int, boxes[i][0])), txtFont, txtSize, rectCol, txtTh)
                coords.append([(boxes[i][0][0] + boxes[i][1][0]) / 2., boxes[i][1][1]])

        return img, np.array(coords)


path = os.path.join(os.path.abspath('.'), "PlayerDetection/Res/Videos/RugbyGame_0.mp4")
if not os.path.exists(path):
    print(f"The path {path} does not exist")
    raise ValueError(path)

beginning = time.time()
fvs = FileVideoStream(path).start()
FullQueue.wait()
FullQueue.clear()

out_path = os.path.join(os.path.abspath('.'), "PlayerDetection/Res/Videos/PlayerEstimate_0.mp4")
if os.path.exists(out_path):
    print(f"The path {path} already exists")
    fvs.halt()
    raise ValueError(out_path)

fourcc = cv.VideoWriter.fourcc(*"MP4V")
out = cv.VideoWriter(out_path, fourcc, 20.0, (1920, 1080))

fps = FPS().start()

i = 0
while fvs.more() or not fvs.halted:
    if fvs.more():
        frame = fvs.read()
        frame = cv.resize(frame, (1920, 1080), interpolation=cv.INTER_AREA)
        out.write(frame)
        print(i)
        i += 1
        fps.update()
    else:
        FullQueue.clear()
        FullQueue.wait()


fps.stop()
end = time.time()
elapsed = end - beginning
print(f"elapsed writing time: {fps.elapsed():.2f}")
print(f"time entire processing has taken: {elapsed:.2f}")
print(f"approximate FPS: {fps.fps():.2f}")
print(f"frames in total: {i}")

out.release()
cv.destroyAllWindows()

fvs.halt()