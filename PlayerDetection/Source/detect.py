import torch
import torchvision
from torchvision import transforms as t
import cv2
import colors
import threading
from PIL import Image
import matplotlib.pyplot as plt
import colorsys as cs


def get_color(i, cropped):
    cols = colors.detect_color(cropped, field_color)
    thread_results[i] = cols[0] if len(cols) > 0 else (0, 0, 0)


def detect_colors(image, data:tuple, ratio):
    global thread_results
    objects = {}
    boxes, classes, scores = data

    threads = []
    for i, (score, label, box) in enumerate(zip(scores, classes, boxes)):
        if label == "person":
            box = [tuple(map(int, box[0])), tuple(map(int, box[1]))]
            [(x1, y1), (x2, y2)] = box
            box = (x1, y1, x2, y2)
            cropped = image[y1:y2, x1:x2]
            ch, cw, _ = cropped.shape
            cropped = cv2.resize(cropped, (int(cw / ratio), int(ch / ratio)))
            cv2.imshow('Y1', cropped)
            # Create thread object with get_color target function
            threads.append(threading.Thread(target=get_color, args=(i, cropped)))
            objects[i] = {"object": "Person",
                          "score": round(score.item(), 3),
                          "location": box}

    thread_results = [(0, 0, 0)] * (max(objects.keys()) + 1)
    # Start all threads
    [t.start() for t in threads]
    print(f"{len(threads)} threads started", sep='\n')
    # Join all threads to the parent thread
    [t.join() for t in threads]
    print(f"{len(threads)} threads finished", sep='\n')

    # Distribute detected colors for objects
    for i in objects.keys():
        objects[i]["color"] = thread_results[i]

    all_colors = [color["color"] for color in objects.values()]

    # Get colour groups
    groups = colors.get_ranged_groups(all_colors, groups_color_filers)

    # Distribute objects by groups
    for key, value in objects.items():
        for k, v in groups.items():
            if value["color"] in v:
                objects[key]["team"] = k

    return image, objects


def draw_boxes(img, objects, ratio):
    for key, value in objects.items():
        try:
            x1, y1, x2, y2 = [int(i * ratio) for i in value['location']]
            # cv2.rectangle(image,(x1, y1),(x2, y2), (255, 255, 0), 5)   # highlight color detection mistakes
            cv2.rectangle(img, (x1, y1), (x2, y2), rect_color[value['team']], 5)
            cv2.putText(img, str(value['team']), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
            # Print objects parameters to the STDOUT
            print(
                f"Detected {value['object']} #{key:<3} "
                f"with score {value['score']:<5}, "
                f"team: {value['team']}, "
                f"color: {str(value['color']):<18}, "
                f"at location {value['location']} "
            )
        except KeyError as e:
            if str(e) == "'team'":
                print(f"Color {value['color']} doesnt match any team color range")
    return img


def GetPred(Img:cv2.Mat, Thr:float):
    img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
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
    return pred_boxes, pred_class, pred_score

def plot_pts(objects):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("Saturation")
    ax.set_ylabel("Hue")
    ax.set_zlabel("Value")

    xs, ys, zs = [], [], []

    for i in objects.keys():
        rgb_col = objects[i]["color"]
        hsv_col = cs.rgb_to_hsv(rgb_col[0] / 255., rgb_col[1] / 255., rgb_col[2] / 255.)
        (y, x, z) = hsv_col
        xs.append(x)
        ys.append(y)
        zs.append(z)

    ax.stem(xs, ys, zs)

    plt.show()



if __name__ == '__main__':

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")


    COCO_CategoryNames = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Define video source, file or stream
    cap = cv2.VideoCapture("PlayerDetection/Res/Videos/RugbyGameSample_0.mp4")

    # Create team groups and adjust their color ranges. Each group can contain more than one color range
    # Required HSV parameters for each filter:
    # rgb_color, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value
    groups_color_filers = {"R": ((110, 150, 0.1, 0.7, 0, 1),),
                           "T1": ((300, 360, 0.1, 0.6, 0.1, 0.6),(0, 50, 0.1, 0.7, 0.1, 0.6),),
                           "T2": ((20, 100, 0.3, 1., 0.1, 1.),)}

    # Adjust playground field color range according to the input colors
    field_color = (40, 150, 0.15, 1, 0.3, 0.8)

    # Define colors for obj rectangles
    rect_color = {'R': (0, 255, 255),
                  'T1': (255, 0, 0),
                  'T2': (0, 0, 255),
                  'G': (255, 0, 255)}

    # load model
    model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    model.to(device)
    model.eval()

    # Process every 3rd frame. Used if high resolution causes slow motion
    counter, target = 0, 2

    # Processing video source in the loop
    while True:
        if counter == target:
            ret, frame = cap.read()
            counter = 0
            if not ret:
                print('Loop')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Original frame resized for better visibility
            orig_frame = cv2.resize(frame, (1920, 1080))
            # Prepare low res frame for AI model recognition
            frame = cv2.resize(frame, (640, 360))

            # calculate aspect ratio for coordinates
            h0, w0, = orig_frame.shape[:2]
            h, w, = frame.shape[:2]
            aspect_ratio = 1
            if w0 / w == h0 / h:
                aspect_ratio = w0 / w

            # perform inference
            (boxes, classes, scores) = GetPred(frame, 0.5)


            # processing objects and show result
            image, obj = detect_colors(frame, (boxes, classes, scores), 1)
            out_image = draw_boxes(orig_frame, obj, aspect_ratio)
            plot_pts(obj)
            cv2.imshow('Faster_R_CNN', out_image)
        else:
            ret = cap.grab()
            counter += 1
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()