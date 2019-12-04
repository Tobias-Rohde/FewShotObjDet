import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import cv2
from yolov3_tiny import *
from yolov3_yodo import *
from yolo import read_darknet_conv_weights
from keras.utils import plot_model
import data
import detection
import matplotlib.pyplot as plt
import numpy as np
from time import time

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def xy_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred[:, 0])


def wh_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred[:, 1])


def obj_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred[:, 2])


def class_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred[:, 3])


def nop_loss(y_true, y_pred):
    # Just sum up everything. At this point y_pred has shape (Batch x 4), where the 4 elements
    # are the individual losses.
    return tf.reduce_sum(y_pred)


def predict(yolo_network_info, model, image, labels, preprocess=True, class_thresh=0.5):
    if preprocess:
        img_preprocess, _, _ = data.letterbox(yolo_network_info, image)
        img_preprocess = img_preprocess[None, :, :, ::-1]  # Need to reverse, since opencv returns BGR
    else:
        img_preprocess = image.copy()
        image = (image[0, ..., ::-1] * 255).astype(np.uint8).copy()

    # Prediction code
    outputs = model.predict(img_preprocess)

    height, width = image.shape[:2]

    detections = detection.get_detections(yolo_network_info, outputs, width, height, 0.5, class_thresh)
    detections = detection.non_max_suppression(yolo_network_info, detections, 0.45, 0.9)

    # Calibrated for 640x360
    line_width = 2
    scale = 0.5

    for det in detections:
        label = "%s (%.2f, %.2f)" % (
            labels[np.argmax(det.class_probs)], det.objectness, np.max(det.class_probs) / det.objectness)

        s, b = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, scale, 1)
        s = s[1] + b

        box = det.box
        top_left = int(np.clip(box.x - box.w / 2.0, s, width - line_width)), \
                   int(np.clip(box.y - box.h / 2.0, 0, height - line_width))
        bottom_right = int(np.clip(box.x + box.w / 2.0, line_width, width - line_width)), \
                       int(np.clip(box.y + box.h / 2.0, line_width, height - line_width))

        cv2.putText(image, label, (top_left[0], top_left[1] - b), cv2.FONT_HERSHEY_TRIPLEX, scale, (0, 0, 255),
                    thickness=1)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), line_width)

    return image


# For debugging. This is the same as my iou tf method, just using np
def iou_np(true, pred):
    def overlap_np(c_1, s_1, c_2, s_2):
        start = np.maximum(c_1 - s_1 / 2.0, c_2 - s_2 / 2.0)
        end = np.minimum(c_1 + s_1 / 2.0, c_2 + s_2 / 2.0)

        # If there is no overlap, this will be negative, so make it 0
        return np.maximum(end - start, 0)

    # In: true = (N x M x 4) pred = (N x H x W x 3 x 85)
    # Out: (N x H x W x 3 x M)

    # Insert degenerate dimensions to allow for
    # np broadcasting
    true = np.reshape(true, (-1, 1, 1, 1, true.shape[1], 4))
    # true = (N x 1 x 1 x 1 x M x 4)
    pred = pred[..., np.newaxis, :4]
    # pred = (N x H x W x 3 x 1 x 4)

    intersect_w = overlap_np(true[..., 0], true[..., 2], pred[..., 0], pred[..., 2])
    intersect_h = overlap_np(true[..., 1], true[..., 3], pred[..., 1], pred[..., 3])
    intersection = intersect_w * intersect_h

    # Inclusion-Exclusion formula
    union = true[..., 2] * true[..., 3] + pred[..., 2] * pred[..., 3] - intersection

    return intersection / union


TRAIN = True
TEST_IMAGE = "pc.jpg"


def main():
    model, _ = create_yolov3_tiny()
    read_darknet_conv_weights(model, r"data/yolov3-tiny.weights")
    model, model_train = build_from_yolov3_tiny(model)

    # plot_model(model_train, "model_train.png", expand_nested=True, show_shapes=True)
    # model_train.summary()

    train_generator = data.ArtificalYOLODataGenerator(YOLOV3_YODO_INFO, "data/train_small/train/",
                                                      "data/logos/", 64, False)
    val_generator = data.ArtificalYOLODataGenerator(YOLOV3_YODO_INFO, "data/train_small/val/",
                                                    "data/logos/", 500, True)
    val_data = val_generator[0]

    if TRAIN:
        optimizer = Adam(lr=0.002, decay=0.0005)
        model_train.compile(loss=nop_loss, optimizer=optimizer, metrics=[xy_loss, wh_loss, obj_loss, class_loss])

        # Code for comparing models

        # print(model_train.metrics_names)
        #
        # for i in range(100):
        #     val_data = val_generator[0]
        #     loss_dict = {}
        #     for f in os.listdir("data/checkpoints/"):
        #         if f.endswith(".hdf5"):
        #             model_train.load_weights(os.path.join("data/checkpoints/", f))
        #             l = model_train.evaluate(val_data[0], val_data[1], verbose=0)[0]
        #             loss_dict[f] = l
        #     print(sorted(loss_dict.items(), key=lambda x: x[1])[:3])

        # Code for analyzing what the network gets wrong
        # print(model_train.metrics_names)
        # np.set_printoptions(precision=2, suppress=True)
        # model_train.load_weights(os.path.join("data/checkpoints/", "dollar_weights.hdf5"))
        # for val_data in val_generator:
        #     for j in range(len(val_data[0][0])):
        #         single_x = [val_data[0][0][j][np.newaxis, ...], val_data[0][1][j][np.newaxis, ...],
        #                     val_data[0][2][j][np.newaxis, ...], val_data[0][3][j][np.newaxis, ...]]
        #         single_y = [val_data[1][0][j][np.newaxis, ...], val_data[1][1][j][np.newaxis, ...]]
        #
        #         img = predict(YOLOV3_YODO_INFO, model, single_x[0], ["1$", "10$", "2$", "20$", "5$"], preprocess=False)
        #         l = np.array(model_train.evaluate(single_x, single_y, verbose=0))
        #         print(l)
        #         if l[0] >= 0.5:
        #             cv2.imshow("pred", img)
        #             cv2.waitKey(0)
        #             debug=True
        #             predict(YOLOV3_YODO_INFO, model, single_x[0], ["1$", "10$", "2$", "20$", "5$"], preprocess=False)

        lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=7.8125e-6)
        save_weight_callback = ModelCheckpoint("data/checkpoints/logo.{epoch:02d}-{val_loss:.2f}.hdf5",
                                               monitor="val_loss", save_best_only=True, verbose=1, mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", patience=15)
        history = model_train.fit_generator(train_generator, epochs=128, verbose=2, shuffle=False,
                                            callbacks=[save_weight_callback, lr_callback, early_stopping],
                                            validation_data=val_data)

        # Plot training & validation loss values

        def plot_losses(prefix, title):
            loss_0 = prefix + "yolo_0_loss_"
            loss_1 = prefix + "yolo_1_loss_"
            xy = np.array(history.history[loss_0 + "xy_loss"]) + \
                 np.array(history.history[loss_1 + "xy_loss"])

            wh = np.array(history.history[loss_0 + "wh_loss"]) + \
                 np.array(history.history[loss_1 + "wh_loss"])

            obj = np.array(history.history[loss_0 + "obj_loss"]) + \
                  np.array(history.history[loss_1 + "obj_loss"])

            classl = np.array(history.history[loss_0 + "class_loss"]) + \
                     np.array(history.history[loss_1 + "class_loss"])

            plt.plot(xy, label="xy")
            plt.plot(wh, label="wh")
            plt.plot(obj, label="obj")
            plt.plot(classl, label="class")
            plt.plot(history.history[prefix + "loss"], label="sum")
            plt.title(title)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.ylim(0, 4)
            plt.show()
            plt.clf()

        plot_losses("", "YOLO Train Losses")
        plot_losses("val_", "YOLO Val Losses")

        print(np.min(history.history["val_loss"]))

    # eval
    image = cv2.imread(TEST_IMAGE)
    image = predict(YOLOV3_TINY_INFO, model, image, data.COCO_LABELS)
    cv2.imwrite("detect-" + TEST_IMAGE, image)


def main_test_cam():
    cap = cv2.VideoCapture(0)

    model, _ = create_yolov3_tiny()
    model, model_train = build_from_yolov3_tiny(model)
    model.summary()
    # read_darknet_conv_weights(model_train, r"data/yolov3-tiny.weights")
    model.load_weights("data/checkpoints/logo.73-1.02.hdf5")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        frame = predict(YOLOV3_YODO_INFO, model, frame, ["nvidia", "tf", "uw"])

        cv2.imshow("YOLO", frame)
        key = cv2.waitKey(1)
        if key == ord(" "):
            cv2.imwrite("data/screenshots/frame-%i.jpg" % frame_idx, frame)
        if key == ord("b"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# Demo for poster session.
# Since loading takes long on the jetson nano, I made it so that I can switch between models and weights
# at runtime.
def main_demo():
    original, _ = create_yolov3_tiny()
    model, _ = build_from_yolov3_tiny(original)

    model_map = {ord("y"): [YOLOV3_TINY_INFO, data.COCO_LABELS, "best/yolov3.hdf5"],
                 ord("c"): [YOLOV3_YODO_INFO, ["Cube"], "best/cube.hdf5"],
                 ord("d"): [YOLOV3_YODO_INFO, ["Dollar"], "best/dollar.hdf5"],
                 ord("l"): [YOLOV3_YODO_INFO_3_CLASS, ["NVIDIA", "TF", "UW"], "best/logos.hdf5"],
                 ord("p"): [YOLOV3_YODO_INFO, ["Cone"], "best/pine.hdf5"]}

    class_thresh = 0.2
    current_model = model
    current_info = model_map[ord("c")][0]
    current_labels = model_map[ord("c")][1]
    model.load_weights(model_map[ord("c")][2])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        key = cv2.waitKey(1)
        if key in model_map:
            class_thresh = 0.5
            new_info, current_labels, weight_path = model_map[key]
            print("Switching to model %s" % weight_path)
            if key == ord("y"):
                current_model = original
            else:
                class_thresh = 0.2
                if new_info.classes != current_info.classes:
                    current_model, _ = build_from_yolov3_tiny(original, new_info)

            current_info = new_info
            current_model.load_weights(weight_path)
        if key == ord("b"):
            break

        frame = predict(current_info, current_model, frame, current_labels, True, class_thresh)
        cv2.imshow("YOLO", frame)

    cap.release()
    cv2.destroyAllWindows()


def main_benchmark():
    img = np.zeros((1, 416, 416, 3), dtype=np.float32)

    model, _ = create_yolov3_tiny()
    model, _ = build_from_yolov3_tiny(model)
    model.summary()
    model.load_weights("data/checkpoints/logo.73-1.02.hdf5")

    # Warm up
    for i in range(10):
        model.predict(img)

    start_time = time()
    for i in range(10000):
        model.predict(img)
    print(10000 / (time() - start_time))


if __name__ == "__main__":
    main_demo()
