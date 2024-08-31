import cv2 as cv
import threading
import ColorChecking as colors


def get_color(i, cropped):
    thread_results[i] = colors.detect_color(cropped, field_color)[0]

def detect_colors(img, boxes, ratio):
    global thread_results
    objects = {}
    
    threads = []
    for i, box in enumerate(boxes):
        box = [tuple(map(int, boxes[i][0])), tuple(map(int, boxes[i][1]))]
        x0, y0 = box[0]
        x1, y1 = box[1]
        cropped = img[y0:y1, x0:x1]
        ch, cw, _ = cropped.shape
        cropped = cv.resize(cropped, (int(cw / ratio), int(ch / ratio)))
        threads.append(threading.Thread(target=get_color, args=(i, cropped)))
        objects[i] = {"object": "Person",
                    "score": round(score.item(), 3),}