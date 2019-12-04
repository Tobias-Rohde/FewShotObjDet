from keras.layers import Conv2D, Layer, BatchNormalization, LeakyReLU
from keras import Sequential
import tensorflow as tf
import numpy as np
import struct


class YOLONetworkInfo:
    def __init__(self, input_width, input_height, num_yolo_layers, layer_sizes, anchors, masks, classes,
                 max_boxes_per_img, ignore_threshold):
        # Size of input images
        self.input_size = np.array([input_width, input_height])

        # Number of YOLO layers
        self.num_yolo_layers = num_yolo_layers

        # Sizes of feature map of YOLO layers
        self.layer_sizes = np.array(layer_sizes)

        # Anchor boxes
        self.anchors = np.array(anchors)

        # Masks for indexing into anchors for corresponding yolo layers
        self.masks = np.array(masks)

        # Number of box predictors per cell, equivalent to len(MASKS[i]) or len(ANCHORS) / len(MASKS)
        self.boxes_per_cell = len(self.anchors) // len(self.masks)

        # Number of classes to predict
        self.classes = classes

        # Maximum number of boxes an image can be annotated with, only used for training
        self.max_boxes_per_img = max_boxes_per_img

        # Threshold for ignoring loss from large IOUs whose box predictors
        # are not responsible for the given truth
        self.ignore_threshold = ignore_threshold


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, stride, conv_idx=-1, last=False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        # With branches in the computation graph such as yolov3 (2 outputs),
        # Keras can reorder layers when creating the model. In order to load the weights from
        # darknet, we need to keep track of the order of the conv blocks.
        self.conv_idx = conv_idx
        self.last = last

        self.network = None

    def build(self, input_shape):
        layers = [Conv2D(self.filters, self.kernel_size, strides=self.stride, padding="same",
                         input_shape=input_shape[1:], use_bias=self.last)]
        if not self.last:
            layers.append(BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5))
            layers.append(LeakyReLU(0.1))
        self.network = Sequential(layers=layers)

    def compute_output_shape(self, input_shape):
        return self.network.get_layer(index=0).compute_output_shape(input_shape)

    def call(self, inputs, **kwargs):
        return self.network(inputs)


class YOLO(Layer):
    def __init__(self, yolo_net_info: YOLONetworkInfo, **kwargs):
        super(YOLO, self).__init__(**kwargs)

        # Define properties, but initialize in build
        self.yolo_net_info = yolo_net_info
        self.net_input_size = None
        self.anchors = None

    def build(self, input_shape):
        layer_idx = next(i for i, s in enumerate(self.yolo_net_info.layer_sizes) if s == input_shape[1])
        self.net_input_size = tf.constant(self.yolo_net_info.input_size, dtype=tf.float32)
        self.anchors = tf.constant(self.yolo_net_info.anchors[self.yolo_net_info.masks[layer_idx]], dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        # t_x and t_y
        t_xy = tf.sigmoid(inputs[..., :2])

        # t_w and t_h
        t_wh = tf.exp(inputs[..., 2:4])

        # p_o and p_c's
        probabilities = tf.sigmoid(inputs[..., 4:])

        size = inputs.shape[1]

        # Add cell coordinates (i.e indices) to the predicted center offsets.
        c_x = tf.tile(tf.range(size, dtype=tf.float32)[tf.newaxis, :], [size, 1])
        c_y = tf.transpose(c_x)
        c_xy = tf.expand_dims(tf.stack([c_x, c_y], -1), 2)
        b_xy = t_xy + c_xy

        # Currently b_xy is at the feature scale, need to multiply by net size / feature size  to scale it to
        # the input image size
        b_xy = b_xy * (self.net_input_size / size)

        # Multiply with anchors p_w = self.anchors[0], p_h = self.anchors[1] (the anchors essentially perform the
        # above scale transformation)
        b_wh = t_wh * self.anchors

        # We also return the raw inputs (after sigmoid), since we need those for the loss calculation.
        # The alternative would be to again subtract the grid and divide by anchor and log,
        # but that's silly since we have this info
        return tf.concat([b_xy, b_wh, probabilities], -1)


def iou_tf(true, pred):
    def overlap_tf(c_1, s_1, c_2, s_2):
        start = tf.maximum(c_1 - s_1 / 2.0, c_2 - s_2 / 2.0)
        end = tf.minimum(c_1 + s_1 / 2.0, c_2 + s_2 / 2.0)

        # If there is no overlap, this will be negative, so make it 0
        return tf.maximum(end - start, 0)

    # In: true = (N x M x 4) pred = (N x H x W x 3 x 85)
    # Out: (N x H x W x 3 x M)

    # Insert degenerate dimensions to allow for
    # tf broadcasting
    true = tf.reshape(true, (-1, 1, 1, 1, true.shape[1], 4))
    # true = (N x 1 x 1 x 1 x M x 4)
    pred = pred[..., tf.newaxis, :4]
    # pred = (N x H x W x 3 x 1 x 4)

    intersect_w = overlap_tf(true[..., 0], true[..., 2], pred[..., 0], pred[..., 2])
    intersect_h = overlap_tf(true[..., 1], true[..., 3], pred[..., 1], pred[..., 3])
    intersection = intersect_w * intersect_h

    # Inclusion-Exclusion formula
    union = true[..., 2] * true[..., 3] + pred[..., 2] * pred[..., 3] - intersection

    return intersection / union


class YOLOLossLayer(Layer):
    def __init__(self, yolo_net_info: YOLONetworkInfo, **kwargs):
        super(YOLOLossLayer, self).__init__(**kwargs)

        self.yolo_net_info = yolo_net_info

        # Define properties, but initialize in build
        self.net_input_size = None
        self.anchors = None
        self.ignore_threshold = None

    def build(self, input_shape):
        layer_idx = next(i for i, s in enumerate(self.yolo_net_info.layer_sizes) if s == input_shape[0][1])
        self.net_input_size = tf.constant(self.yolo_net_info.input_size, dtype=tf.float32)
        self.anchors = tf.constant(self.yolo_net_info.anchors[self.yolo_net_info.masks[layer_idx]], dtype=tf.float32)
        self.ignore_threshold = self.yolo_net_info.ignore_threshold

    def call(self, inputs, **kwargs):
        yolo_pred, yolo_truth, yolo_boxes = inputs

        # shapes (N x H x W x 3 x 85)

        # This was not described in the paper, but we are scaling both box losses according to the size
        # of the box. If the box covers the entire image, we have 2 - 1 = 1 so we don't change the scale.
        # if the box size is 0 < s < 1, 1 < 2 - s < 2, so we scale up the loss, thus penalizing more for small
        # boxes
        scale = tf.expand_dims(2 - yolo_truth[..., 2] / self.net_input_size[0]
                               * yolo_truth[..., 3] / self.net_input_size[1], axis=-1)

        obj_mask = yolo_truth[..., 4, tf.newaxis]
        reduce_no_batch = tf.range(1, tf.rank(yolo_pred))

        # Calculate loss for x, y offsets. The tensor values are scaled to the network input size, so
        # rescale to grid size
        xy_loss = tf.reduce_sum(
            tf.square(scale * obj_mask * (yolo_truth[..., :2] - yolo_pred[..., :2])
                      / self.net_input_size * tf.cast(yolo_truth.shape[1:3], tf.float32)),
            reduce_no_batch)[..., tf.newaxis]

        # Calculate loss for w, h values. Note that b = p e^t, which is the value we get from the tensors.
        # However we want to calculate loss based on t, so need to solve for t as
        # t = log(b/p). (p are the anchors)
        wh_delta = tf.math.log(tf.clip_by_value(yolo_truth[..., 2:4] / self.anchors, 1e-10, self.net_input_size[0])) - \
                   tf.math.log(yolo_pred[..., 2:4] / self.anchors)
        wh_loss = tf.reduce_sum(tf.square(scale * obj_mask * wh_delta), reduce_no_batch)[..., tf.newaxis]

        ious = iou_tf(yolo_boxes, yolo_pred)

        # For each cell, get the maximum iou out of the different 3 anchor boxes
        ious = tf.reduce_max(ious, axis=-1)
        # If IOU is small or there is an object, keep the objectiveness and penalize for it
        obj_iou_mask = 1 - (1 - yolo_truth[..., 4]) * (tf.cast(ious > self.ignore_threshold, tf.float32))
        obj_loss = tf.reduce_sum(tf.square(obj_iou_mask * (yolo_truth[..., 4] - yolo_pred[..., 4])),
                                 reduce_no_batch[:-1])[..., tf.newaxis]

        class_loss = tf.reduce_sum(tf.square(obj_mask * (yolo_truth[..., 5:] - yolo_pred[..., 5:])),
                                   reduce_no_batch)[..., tf.newaxis]

        return 0.5 * tf.concat([xy_loss, wh_loss, obj_loss, class_loss], axis=1)


def read_darknet_conv_weights(model, file_path):
    with open(file_path, "rb") as f:
        # major, minor, rev, seen. We don't care about any of those but just want to skip them
        f.read(4 * 3 + 8)

        conv_blocks = [l for l in model.layers if type(l).__name__ == ConvBlock.__name__]
        # Sort ascending by conv index so that we read weights in the correct order
        conv_blocks = sorted(conv_blocks, key=lambda l: l.conv_idx)

        for conv_block in conv_blocks:
            # The first layer in ConvBlock is the conv layer
            conv = conv_block.network.get_layer(index=0)
            print("%s -> %s" % (str(conv.input_shape), str(conv.compute_output_shape(conv.input_shape))))

            conv_weights_shape = conv.weights[0].shape
            kernel_x = conv_weights_shape[0]
            kernel_y = conv_weights_shape[1]
            filter_in = conv_weights_shape[2]
            filter_out = conv_weights_shape[3]
            weights_size = tf.size(conv.weights[0]).numpy()  # Or just product of the above

            biases = np.array(struct.unpack("<%if" % filter_out, f.read(4 * filter_out)))

            # A conv block right before a yolo layer does not have batch norm
            if not conv_block.last:
                # The first layer in ConvBlock is the batch norm layer, if existent
                bn = conv_block.network.get_layer(index=1)

                scales = np.array(struct.unpack("<%if" % filter_out, f.read(4 * filter_out)))
                rolling_mean = np.array(struct.unpack("<%if" % filter_out, f.read(4 * filter_out)))
                rolling_variance = np.array(struct.unpack("<%if" % filter_out, f.read(4 * filter_out)))

                # gamma, beta, moving_mean, moving_average
                bn.set_weights([scales, biases, rolling_mean, rolling_variance])

            weights = np.array(struct.unpack("<%if" % weights_size, f.read(4 * weights_size)))
            # Darknet stores (out, in, kx, ky) but keras wants (kx, ky, in, out)
            weights = np.reshape(weights, (filter_out, filter_in, kernel_x, kernel_y))
            weights = np.transpose(weights, axes=(2, 3, 1, 0))

            conv_weights = [weights]
            if conv_block.last:
                conv_weights.append(biases)
            conv.set_weights(conv_weights)
