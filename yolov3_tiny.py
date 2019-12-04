from keras.layers import MaxPool2D, Input, UpSampling2D, Concatenate, Reshape
from yolo import ConvBlock, YOLO, YOLOLossLayer, YOLONetworkInfo
from keras import Model

YOLOV3_TINY_INFO = YOLONetworkInfo(416, 416, 2, [13, 26],
                                   [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
                                   [[3, 4, 5], [0, 1, 2]], 80, 10, 0.7)


def create_yolov3_tiny():
    img_input = Input(shape=(*YOLOV3_TINY_INFO.input_size, 3), name="img_input")

    x = ConvBlock(16, 3, 1, conv_idx=0)(img_input)  # 416
    x = MaxPool2D(2, 2)(x)

    x = ConvBlock(32, 3, 1, conv_idx=1)(x)  # 208
    x = MaxPool2D(2, 2)(x)

    x = ConvBlock(64, 3, 1, conv_idx=2)(x)  # 104
    x = MaxPool2D(2, 2)(x)

    x = ConvBlock(128, 3, 1, conv_idx=3)(x)  # 52
    x = MaxPool2D(2, 2)(x)

    x_route_0 = ConvBlock(256, 3, 1, conv_idx=4)(x)  # 26
    x = MaxPool2D(2, 2)(x_route_0)

    x = ConvBlock(512, 3, 1, conv_idx=5)(x)  # 13
    x = MaxPool2D(2, 1, padding="same")(x)

    x = ConvBlock(1024, 3, 1, conv_idx=6)(x)  # 13

    x_route_1 = ConvBlock(256, 1, 1, conv_idx=7)(x)

    # First predict boxes at smaller scale (13)
    x = ConvBlock(512, 3, 1, conv_idx=8, name="pre_yolo_0_conv")(x_route_1)
    x = ConvBlock(YOLOV3_TINY_INFO.boxes_per_cell * (5 + YOLOV3_TINY_INFO.classes), 1, 1, conv_idx=9,
                  last=True, name="yolo_0_conv")(x)
    x = Reshape(x.shape[1:-1] + (YOLOV3_TINY_INFO.boxes_per_cell, 5 + YOLOV3_TINY_INFO.classes))(x)

    yolo_truth_boxes = Input(shape=(YOLOV3_TINY_INFO.max_boxes_per_img, 4), name="yolo_truth_boxes")

    yolo_0 = YOLO(YOLOV3_TINY_INFO, name="yolo_0")(x)
    yolo_0_truth = Input(shape=yolo_0.shape[1:], name="yolo_0_truth")
    yolo_0_loss = YOLOLossLayer(YOLOV3_TINY_INFO, name="yolo_0_loss")(
        [yolo_0, yolo_0_truth, yolo_truth_boxes])

    # Secondly predict boxes at larger scale
    x = ConvBlock(128, 1, 1, conv_idx=10)(x_route_1)  # 13
    x = UpSampling2D(interpolation="bilinear")(x)
    x = Concatenate(axis=3)([x, x_route_0])  # 26
    x = ConvBlock(256, 3, 1, conv_idx=11, name="pre_yolo_1_conv")(x)
    x = ConvBlock(YOLOV3_TINY_INFO.boxes_per_cell * (5 + YOLOV3_TINY_INFO.classes), 1, 1, conv_idx=12,
                  last=True, name="yolo_1_conv")(x)
    x = Reshape(x.shape[1:-1] + (YOLOV3_TINY_INFO.boxes_per_cell, 5 + YOLOV3_TINY_INFO.classes))(x)

    yolo_1 = YOLO(YOLOV3_TINY_INFO, name="yolo_1")(x)
    yolo_1_truth = Input(shape=yolo_1.shape[1:], name="yolo_1_truth")
    yolo_1_loss = YOLOLossLayer(YOLOV3_TINY_INFO, name="yolo_1_loss")(
        [yolo_1, yolo_1_truth, yolo_truth_boxes])

    inference = Model(inputs=img_input, outputs=[yolo_0, yolo_1])
    train = Model(inputs=[img_input, yolo_0_truth, yolo_1_truth, yolo_truth_boxes], outputs=[yolo_0_loss, yolo_1_loss])

    return inference, train
