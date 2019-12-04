from yolov3_tiny import *
from data import Box, letterbox_size
import numpy as np

class Detection:
    def __init__(self, box=None, objectness=0, class_probs=None):
        self.box = box

        self.objectness = objectness
        self.class_probs = class_probs


def non_max_suppression(yolo_network_info: YOLONetworkInfo, detections, iou_threshold, iomina_threshold):
    for c in range(yolo_network_info.classes):
        # Sort detections by decreasing probability for this class
        detections = sorted(detections, key=lambda de: de.class_probs[c], reverse=True)

        for i in range(len(detections)):
            det = detections[i]
            # If the class has 0 probability, we don't care about the box and all following detections
            # will also have 0 probability for this class since it is sorted
            #if det.class_probs[c] == 0:
            #    break

            # "Remove" all predictions with lesser probability that have a smaller iou by making
            # their class probability 0
            for d in detections[i + 1:]:
                if d.box.iou(det.box) > iou_threshold or d.box.iomina(det.box) > iomina_threshold:
                    d.class_probs[c] = 0

    # Filter out detections who have been suppressed, i.e all their class probabilities have been made 0
    return [d for d in detections if not np.array_equiv(d.class_probs, 0)]


def get_detections(yolo_network_info: YOLONetworkInfo, model_outputs, img_w, img_h, objectness_threshold,
                   class_threshold):
    detections = []

    for n in range(len(model_outputs)):
        out = model_outputs[n][0]
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                for i in range(out.shape[2]):
                    objectness = out[x, y, i, 4]
                    if objectness <= objectness_threshold:
                        continue

                    # Calculate P(Obj /\ Class) = P(Class | Obj) P(Obj)
                    class_probs = out[x, y, i, 5:] * objectness
                    class_probs[class_probs <= class_threshold] = 0

                    det = Detection(Box(*out[x, y, i, :4]), objectness, class_probs)

                    # If we don't have confidence in any classes, discard prediction
                    if np.array_equiv(det.class_probs, 0):
                        continue

                    detections.append(det)

                    # Since the image was letterboxed, we need to account for the size of the gray border
                    # and shift the x, y coordinates accordingly. We also need to scale the boxes back to
                    # the original input dimensions
                    new_w, new_h = letterbox_size(yolo_network_info, img_w, img_h)
                    det.box = det.box.offset((yolo_network_info.input_size[0] - new_w) / 2.0,
                                             (yolo_network_info.input_size[1] - new_h) / 2.0) \
                        .rescale(img_w / new_w, img_h / new_h)

    return detections
