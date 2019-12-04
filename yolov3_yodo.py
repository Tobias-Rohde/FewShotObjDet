from keras.layers import Input, Reshape
from yolo import ConvBlock, YOLO, YOLOLossLayer, YOLONetworkInfo
from yolov3_tiny import YOLOV3_TINY_INFO
from keras import Model

# "You only detect one"

YOLOV3_YODO_INFO = YOLONetworkInfo(416, 416, 2, [13, 26],
                                   [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
                                   [[3, 4, 5], [0, 1, 2]], 1, 3, 0.6)

YOLOV3_YODO_INFO_3_CLASS = YOLONetworkInfo(416, 416, 2, [13, 26],
                                   [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
                                   [[3, 4, 5], [0, 1, 2]], 3, 3, 0.6)


def build_from_yolov3_tiny(yolov3_tiny, info=YOLOV3_YODO_INFO):
    pre_yolov3_tiny_0_conv, pre_yolov3_tiny_1_conv = yolov3_tiny.get_layer("pre_yolo_0_conv").output, \
                                                     yolov3_tiny.get_layer("pre_yolo_1_conv").output

    yolo_0_conv = ConvBlock(info.boxes_per_cell * (5 + info.classes), 1, 1, last=True)
    x = yolo_0_conv(pre_yolov3_tiny_0_conv)
    x = Reshape(x.shape[1:-1] + (info.boxes_per_cell, 5 + info.classes))(x)

    yolo_truth_boxes = Input(shape=(info.max_boxes_per_img, 4), name="yolo_truth_boxes")

    yolo_0 = YOLO(info, name="yolo_0")(x)
    yolo_0_truth = Input(shape=yolo_0.shape[1:], name="yolo_0_truth")
    yolo_0_loss = YOLOLossLayer(info, name="yolo_0_loss")(
        [yolo_0, yolo_0_truth, yolo_truth_boxes])

    yolo_1_conv = ConvBlock(info.boxes_per_cell * (5 + info.classes), 1, 1, last=True)
    x = yolo_1_conv(pre_yolov3_tiny_1_conv)
    x = Reshape(x.shape[1:-1] + (info.boxes_per_cell, 5 + info.classes))(x)

    yolo_1 = YOLO(info, name="yolo_1")(x)
    yolo_1_truth = Input(shape=yolo_1.shape[1:], name="yolo_1_truth")
    yolo_1_loss = YOLOLossLayer(info, name="yolo_1_loss")(
        [yolo_1, yolo_1_truth, yolo_truth_boxes])

    # Copy filter weights for x, y, w, h, objectness filters
    old_yolo_0_conv_weights, old_yolo_1_conv_weights = yolov3_tiny.get_layer("yolo_0_conv").get_weights(), \
                                                       yolov3_tiny.get_layer("yolo_1_conv").get_weights()

    yolo_0_conv_weights = yolo_0_conv.get_weights()
    yolo_1_conv_weights = yolo_1_conv.get_weights()

    box_idx = 0
    for i in range(info.boxes_per_cell):
        new_start_idx = i * (5 + info.classes)
        old_start_idx = box_idx * (5 + YOLOV3_TINY_INFO.classes)

        # Copy filter weights
        yolo_0_conv_weights[0][..., new_start_idx:new_start_idx + 4 + 1] =\
            old_yolo_0_conv_weights[0][..., old_start_idx:old_start_idx + 4 + 1]
        yolo_1_conv_weights[0][..., new_start_idx:new_start_idx + 4 + 1] =\
            old_yolo_1_conv_weights[0][..., old_start_idx:old_start_idx + 4 + 1]
        # Copy filter bias weights
        yolo_0_conv_weights[1][new_start_idx:new_start_idx + 4 + 1] =\
            old_yolo_0_conv_weights[1][old_start_idx:old_start_idx + 4 + 1]
        yolo_1_conv_weights[1][new_start_idx:new_start_idx + 4 + 1] =\
            old_yolo_1_conv_weights[1][old_start_idx:old_start_idx + 4 + 1]

        # This check is currently not necessary, but I figure that if I want to use more box predictors than before,
        # I will just reuse some of the old weights for the additional box predictors several times
        if box_idx < YOLOV3_TINY_INFO.boxes_per_cell - 1:
            box_idx += 1

    yolo_0_conv.set_weights(yolo_0_conv_weights)
    yolo_1_conv.set_weights(yolo_1_conv_weights)

    input = yolov3_tiny.get_layer("img_input").output
    inference = Model(inputs=input, outputs=[yolo_0, yolo_1])
    train = Model(inputs=[input, yolo_0_truth, yolo_1_truth, yolo_truth_boxes], outputs=[yolo_0_loss, yolo_1_loss])

    return inference, train
