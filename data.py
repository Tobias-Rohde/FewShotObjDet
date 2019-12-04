from keras.utils.data_utils import Sequence
from yolo import YOLONetworkInfo
import cv2
import os
import numpy as np

COCO_LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush"]


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    # Offset x,y and return new box
    def offset(self, offset_x, offset_y):
        return Box(self.x - offset_x, self.y - offset_y, self.w, self.h)

    # Rescale x,w,y,w and return new box
    def rescale(self, x_scale, y_scale):
        return Box(self.x * x_scale, self.y * y_scale, self.w * x_scale, self.h * y_scale)

    @staticmethod
    def overlap(c_1, s_1, c_2, s_2):
        start = max(c_1 - s_1 / 2, c_2 - s_2 / 2)
        end = min(c_1 + s_1 / 2, c_2 + s_2 / 2)

        # If negative, there is no overlap and the IOU will become 0
        return max(end - start, 0)

    def intersection(self, other):
        intersection_width = Box.overlap(self.x, self.w, other.x, other.w)
        intersection_height = Box.overlap(self.y, self.h, other.y, other.h)

        return intersection_width * intersection_height

    # Calculate iou with other box
    def iou(self, other):
        intersection = self.intersection(other)

        # Inclusion-Exclusion formula
        union = self.w * self.h + other.w * other.h - intersection

        return intersection / union

    # I noticed that sometimes a smaller box would be almost fully contained in a larger box, resulting in
    # a small iou. Thus also calculate a score for how much the smaller box is contained in the larger box.
    # Call this "intersection over min area" -> iomina
    def iomina(self, other):
        intersection = self.intersection(other)
        return intersection / min(self.w * self.h, other.w * other.h)

    # Return list containing x,y,w,h
    def as_list(self):
        return [self.x, self.y, self.w, self.h]


def letterbox_size(yolo_network_info: YOLONetworkInfo, img_w, img_h):
    # We need to scale either the width or height to the net width or net height, where the other
    # size is less than or equal to the other net size after scaling so that we can fit it into a
    # net width x net height box without distortion
    net_width, net_height = yolo_network_info.input_size

    if img_h / net_height < img_w / net_width:
        # img_h / NET_HEIGHT < img_w / NET_WIDTH
        # img_h * NET_WIDTH / img_w < NET_HEIGHT as wanted
        return net_width, img_h * (net_width / img_w)
    else:
        # img_h / NET_HEIGHT >= img_w / NET_WIDTH
        # NET_WIDTH >= img_w * NET_HEIGHT / img_h as wanted
        return img_w * (net_height / img_h), net_height


# We letter box the image to fit the network instead of simply scaling it down,
# which would cause the image to become distorted
def letterbox(yolo_network_info: YOLONetworkInfo, image):
    # image shape is h, w not w, h, so reverse
    new_w, new_h = letterbox_size(yolo_network_info, *image.shape[1::-1])
    new_w, new_h = int(new_w), int(new_h)

    # new_w, new_h have the same aspect ratio as  the original image, so we can safely resize
    image_resize = cv2.resize(image, (new_w, new_h)) / 255

    # Use gray on border
    result = np.full((*yolo_network_info.input_size, 3), 0.5, dtype=np.float32)

    net_width, net_height = yolo_network_info.input_size
    # Insert image into the gray background such that it is centered in x and y direction
    result[(net_height - new_h) // 2: new_h + (net_height - new_h) // 2,
    (net_width - new_w) // 2: new_w + (net_width - new_w) // 2, :] = image_resize

    return result, new_w, new_h


# Not used anymore. I used this at the start for testing purposes to see if I can get YOLO to overfit
# to some actual data (dog bike car picture), to make sure I got the loss function right
def load_data(yolo_network_info: YOLONetworkInfo, data_path):
    # I store data in the format: path ... x_i y_i w_i h_i id_i ...
    # Where id is zero based and x_i, y_i is the center of the bounding box and coordinates and sizes
    # are scaled to be between 0 and 1.

    with open(data_path, "r") as f:
        yolo_input_images = []
        yolo_truth = [[] for _ in range(yolo_network_info.num_yolo_layers)]
        yolo_boxes = []

        net_width, net_height = yolo_network_info.input_size

        while True:
            line = f.readline()
            if line == "":
                break

            img_data = line.split(" ")
            img_path = img_data[0]
            img = cv2.imread(os.path.join(os.path.dirname(data_path), img_path))

            # Need to reverse, since opencv returns BGR
            img_preprocess, new_w, new_h = letterbox(img)
            img_preprocess = img_preprocess[:, :, ::-1]
            yolo_input_images.append(img_preprocess)

            size_scale = [new_w, new_h, new_w, new_h]

            anchor_boxes = [Box(0, 0, *a) for a in yolo_network_info.anchors]

            for i in range(yolo_network_info.num_yolo_layers):
                yolo_truth[i].append(np.zeros((yolo_network_info.layer_sizes[i], yolo_network_info.layer_sizes[i],
                                               yolo_network_info.boxes_per_cell, 4 + 1 + yolo_network_info.classes),
                                              dtype=np.float32))

            data_all = np.zeros((yolo_network_info.max_boxes_per_img, 4), dtype=np.float32)

            # 5 because x_i y_i w_i h_i id_i
            for i in range(1, len(img_data), 5):
                # Adjust box data to match letterboxed and resized image
                img_box = Box(*(np.array([float(v) for v in img_data[i:i + 4]])) * size_scale)
                img_box = img_box.offset((net_width - new_w) // 2, (net_height - new_h) // 2)
                img_id = int(img_data[i + 4])

                data_all[(i - 1) // 5] = img_box.as_list()

                # Get box at origin
                img_box_o = img_box.offset(img_box.x, img_box.y)
                max_iou_idx = np.argmax([img_box_o.iou(a) for a in anchor_boxes])
                layer_idx = [i for i, indices in enumerate(yolo_network_info.masks) if max_iou_idx in indices][0]
                anchor_index = np.argmax(yolo_network_info.masks[layer_idx] == max_iou_idx)
                size = yolo_network_info.layer_sizes[layer_idx]

                # Determine the cell that is responsible for this prediction, i.e
                # the cell which contains the center of the box
                cell_x_idx = int(img_box.x * size / net_width)
                cell_y_idx = int(img_box.y * size / net_height)

                yolo_truth[layer_idx][-1][cell_y_idx, cell_x_idx, anchor_index, :4] = img_box.as_list()
                yolo_truth[layer_idx][-1][cell_y_idx, cell_x_idx, anchor_index, 4] = 1
                yolo_truth[layer_idx][-1][cell_y_idx, cell_x_idx, anchor_index, 5 + img_id] = 1

            yolo_boxes.append(data_all)

    for i in range(yolo_network_info.num_yolo_layers):
        yolo_truth[i] = np.stack(yolo_truth[i])

    return np.stack(yolo_input_images), yolo_truth, np.stack(yolo_boxes)


# Split data into train test and val folders. Parameters are the
# proportions. Must sum to 1. (Note that test is technically ignored and just
# assumed to be 1-train-val.
def split_data_set(path, train, val, test):
    file_names = os.listdir(path)

    os.mkdir(os.path.join(path, "train"))
    os.mkdir(os.path.join(path, "val"))
    os.mkdir(os.path.join(path, "test"))

    idx = np.random.permutation(len(file_names))
    train_size = int(len(file_names) * train)
    val_size = int(len(file_names) * val)
    # test_size = int(len(file_names) * val)

    for i in idx[:train_size]:
        f = file_names[i]
        os.rename(os.path.join(path, f), os.path.join(path, "train", f))

    for i in idx[train_size:train_size + val_size]:
        f = file_names[i]
        os.rename(os.path.join(path, f), os.path.join(path, "val", f))

    for i in idx[train_size + val_size:]:
        f = file_names[i]
        os.rename(os.path.join(path, f), os.path.join(path, "test", f))


class ArtificalYOLODataGenerator(Sequence):
    def __init__(self, yolo_network_info: YOLONetworkInfo, base_data_path, object_path, batch_size, is_val):
        self.yolo_network_info = yolo_network_info

        self.base_data_path = base_data_path
        self.batch_size = batch_size

        self.base_file_names = os.listdir(base_data_path)
        self.base_data_indices = np.random.permutation(len(self.base_file_names))

        # Just read the objects into memory, since there won't be many
        self.class_labels = os.listdir(object_path)
        # one folder per class
        assert (len(self.class_labels) == yolo_network_info.classes)
        # Read image using cv2.IMREAD_UNCHANGED to keep alpha channel and convert channels to RGBA.
        # Divide by 255.0 so we get 0-1 and floats

        self.object_images = [None] * len(self.class_labels)
        for i, obj_class in enumerate(self.class_labels):
            object_file_names = os.listdir(os.path.join(object_path, obj_class))
            self.object_images[i] = [cv2.cvtColor(cv2.imread(os.path.join(object_path, obj_class, f),
                                     cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
                                     for f in object_file_names]

        self.is_val = is_val

    # This was worth a try, but it did not work as well as expected

    # def preprocess_object_images(self, object_images):
    #     result = [None] * len(object_images)
    #
    #     green_thresholds = np.array([[50, 70], [100, 255], [100, 255]], dtype=np.uint8)
    #
    #     for idx, obj_img in enumerate(object_images):
    #         # Add alpha channel if not there already
    #         if obj_img.shape[-1] == 3:
    #             obj_img = np.pad(obj_img, ((0, 0), (0, 0), (0, 1)), constant_values=255)
    #
    #         # Convert to HSV for easier filtering
    #         obj_img[..., :3] = cv2.cvtColor(obj_img[..., :3], cv2.COLOR_BGR2HSV)
    #
    #         x = obj_img[..., :3]
    #         # Check if each pixels RGB values lie within the green screen range
    #         green_mask = np.logical_and(x >= green_thresholds[..., 0], x <= green_thresholds[..., 1])
    #         # All three color channels must be in the range, so reduce the channel dimension using 'and'
    #         green_mask = np.all(green_mask, axis=-1)
    #         # Set everything green to 0, including alpha channel
    #         obj_img[green_mask] = 0
    #
    #         transparent = obj_img[..., 3] == 0
    #
    #         # Remove transparency from left and right
    #         transparent_x = np.all(transparent, axis=0)
    #         transparent_x_start = np.argmin(transparent_x)
    #         transparent_x_end = np.argmin(transparent_x[::-1])
    #         # Remove transparency from top and bottom
    #         transparent_y = np.all(transparent, axis=1)
    #         transparent_y_start = np.argmin(transparent_y)
    #         transparent_y_end = np.argmin(transparent_y[::-1])
    #
    #         transformed_img = obj_img[transparent_y_start:obj_img.shape[0] - transparent_y_end,
    #                           transparent_x_start:obj_img.shape[1] - transparent_x_end]
    #
    #         greenscreen = obj_img.copy()
    #         greenscreen[..., :3] = cv2.cvtColor(greenscreen[..., :3], cv2.COLOR_HSV2BGR)
    #         cv2.imwrite(str(idx) + "greenscreen.png", greenscreen)
    #
    #         pass


    # OLD:
    # # Resize the object such that its larger side length takes up between an 8th and a 4th of the
    # # base images smaller side length
    # img_min_size = np.minimum(base_w, base_h)
    # obj_min_size, obj_max_size = img_min_size // 10, img_min_size // 2
    # obj_size = np.random.randint(obj_min_size, obj_max_size + 1)
    # old_obj_h, old_obj_w = object_img.shape[:2]
    # # Scale by inverse of larger side so that the larger side length becomes obj_size
    # scale = max(old_obj_h, old_obj_w)
    # # Round can cause the values to go below (or above?) the min and max size, so clip
    # new_obj_w = np.clip(int(old_obj_w / scale * obj_size), obj_min_size, obj_max_size)
    # new_obj_h = np.clip(int(old_obj_h / scale * obj_size), obj_min_size, obj_max_size)
    # object_img = cv2.resize(object_img, (new_obj_w, new_obj_h))

    @staticmethod
    def augment_object_img(object_img, base_w, base_h):
        # Only used this for dollar bill
        # if np.random.randint(0, 8) % 8 == 0:
        #     if np.random.randint(0, 2) % 2 == 0:
        #         object_img = object_img[:, :object_img.shape[1]//2, :]
        #     else:
        #         object_img = object_img[:, object_img.shape[1] // 2:, :]

        # Resize the object such that its larger side length takes up between a 10h and a half of the
        # base images smaller side length
        img_min_size = np.minimum(base_w, base_h)
        obj_min_size, obj_max_size = img_min_size // 8, int(img_min_size / 1.5)

        old_obj_h, old_obj_w = object_img.shape[:2]
        min_len, max_len = min(old_obj_h, old_obj_w), max(old_obj_h, old_obj_w)

        # Determine scale range that won't cause either side length to go below or beyond obj_min_size or obj_max_size
        min_scale, max_scale = obj_min_size / max_len, obj_max_size / max_len

        scale = np.random.uniform(min_scale, max_scale)

        # Round can cause the values to go below (or above?) the min and max size, so clip
        new_obj_w = np.clip(round(old_obj_w * scale), None, obj_max_size)
        new_obj_h = np.clip(round(old_obj_h * scale), None, obj_max_size)

        object_img = cv2.resize(object_img, (new_obj_w, new_obj_h))

        # Randomly flip horizontally. It's fine if we do this often
        # if np.random.randint(0, 2) % 2 == 0:
        #     object_img = cv2.flip(object_img, 1)

        # Randomly shift Hue of HSV values. But don't shift 0 values, so remember where they were
        # and set them back to 0
        zeros = object_img == 0
        # H is in range 0..179, while S and V are in range 0..255. Since we only use H, it is fine to scale
        # the other two incorrectly
        object_img[..., :3] = (cv2.cvtColor((object_img[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 179)\
            .astype(np.float32)
        shift = np.random.uniform(-8 / 179, 8 / 179)
        object_img[..., 0] += shift
        object_img = np.clip(object_img, 0, 1)
        object_img[..., :3] = (cv2.cvtColor((object_img[..., :3] * 179).astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0) \
            .astype(np.float32)
        object_img[zeros] = 0

        # Random gamma correction, found online that one should use a lookup table to make this faster,
        # since then we only have to exponentiate 256 times
        gamma = np.random.uniform(1/3, 1.5)
        lut = ((np.arange(256) / 255) ** (1 / gamma) * 255).astype(np.uint8)
        # Don't want to do gamma correction on the alpha channel
        object_img_noalpha = (object_img[..., :3] * 255).astype(np.uint8)
        object_img[..., :3] = (cv2.LUT(object_img_noalpha, lut) / 255.0).astype(np.float32)

        # Randomly make gray
        if np.random.randint(0, 8) % 8 == 0:
            object_img[..., :3] = np.mean(object_img[..., :3], axis=-1)[..., np.newaxis]

        # Random rotation between 0 and 359
        rot_angle_deg = np.random.randint(0, 360)

        object_center = [object_img.shape[1] // 2, object_img.shape[0] // 2]
        rot_matr = cv2.getRotationMatrix2D(tuple(object_center), rot_angle_deg, 1.0)

        # I don't want pixels to get cut off or filled, so we need to calculate the
        # new width and height after the rotation and figure out the right center to rotate around

        # Idea from: https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
        # First determine the new size after rotating the image
        new_size = np.matmul(np.abs(rot_matr[:2, :2]), [new_obj_w, new_obj_h])
        # By default, the rotation matrix maps the center to itself, but we want to map the center
        # to the center of the new resized image
        rot_matr[:, 2] += new_size / 2 - object_center
        object_img = cv2.warpAffine(object_img, rot_matr, tuple([int(c) for c in new_size]))

        # Some transformations cause alpha channels between 0 < alpha < 1, make them 0.
        # If we make all of those 0, we remove some pixels that have a high alpha channel like 0.9. I don't know
        # how the pixels are "merged" but if we assume that there is some sort of average going on with the surrounding
        # pixels, then if the middle pixel has alpha=1 and the surrounding ones have alpha=0, then the average is
        # 1/9, so use that as a cutoff
        object_img[object_img[..., 3] <= 1 / 9, 3] = 0

        # There is still a problem with the rotations: likely the image does not fill the corners, so there will be
        # lots of transparent space. We can remove this easily

        transparent = object_img[..., 3] == 0

        # Remove transparency from left and right
        transparent_x = np.all(transparent, axis=0)
        transparent_x_start = np.argmin(transparent_x)
        transparent_x_end = np.argmin(transparent_x[::-1])
        # Remove transparency from top and bottom
        transparent_y = np.all(transparent, axis=1)
        transparent_y_start = np.argmin(transparent_y)
        transparent_y_end = np.argmin(transparent_y[::-1])

        transformed_img = object_img[transparent_y_start:object_img.shape[0] - transparent_y_end,
                                     transparent_x_start:object_img.shape[1] - transparent_x_end]

        # Randomly blur to try improve performance with moving camera
        if np.random.randint(0, 8) % 8 == 0:
            # Motion blur
            if np.random.randint(0, 2) % 2 == 0:
                blur_type = np.random.randint(0, 4)
                # Want odd kernel: either 3 or 5
                kernel_size = 2 * np.random.randint(1, 2 + 1) + 1
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

                val = 1 / kernel_size
                if blur_type == 0:  # horizontal
                    kernel[(kernel_size - 1) // 2, :] = val
                elif blur_type == 1:  # vertical
                    kernel[:, (kernel_size - 1) // 2] = val
                elif blur_type == 2:  # diagonal top left to bottom right
                    kernel[np.arange(kernel_size), np.arange(kernel_size)] = val
                elif blur_type == 3:  # diagonal top right to bottom left
                    kernel[np.arange(kernel_size), np.arange(kernel_size - 1, -1, -1)] = val

                transformed_img = cv2.filter2D(transformed_img, -1, kernel)
            # Normal blur
            else:
                # object_img[..., :3] = cv2.GaussianBlur(object_img[..., :3], (3, 3), 0)
                object_img[..., :3] = cv2.blur(object_img[..., :3], (3, 3))

        return transformed_img

    def insert_object_into_image(self, base_img, base_w, base_h):
        insert_count = np.random.randint(1, 3 + 1)

        yolo_truth = [np.zeros((self.yolo_network_info.layer_sizes[i], self.yolo_network_info.layer_sizes[i],
                                self.yolo_network_info.boxes_per_cell, (4 + 1 + self.yolo_network_info.classes)),
                               dtype=np.float32)
                      for i in range(self.yolo_network_info.num_yolo_layers)]

        yolo_boxes = np.zeros((self.yolo_network_info.max_boxes_per_img, 4), dtype=np.float32)
        result_img = base_img.copy()

        j = 0
        while j < insert_count:
            # Randomly choose a class to insert and image from
            class_id = np.random.randint(0, len(self.object_images))
            object_images = self.object_images[class_id]
            # Randomly choose an image from that class
            obj_idx = np.random.randint(0, len(object_images))
            object_img = object_images[obj_idx]

            transformed_img = self.augment_object_img(object_img, base_w, base_h)
            obj_h, obj_w = transformed_img.shape[:2]

            # transformed_img_write = (cv2.cvtColor(transformed_img, cv2.COLOR_RGBA2BGRA) * 255).astype(np.uint8)
            # cv2.imwrite("transformed_img.png", transformed_img_write)

            net_width, net_height = self.yolo_network_info.input_size

            # Size of gray border
            base_w_border = (net_width - base_w) // 2
            base_h_border = (net_height - base_h) // 2

            # Generate random location to insert object into. Make sure the object is fully within the image.
            x_loc = np.random.randint(0, base_w - obj_w) + base_w_border
            y_loc = np.random.randint(0, base_h - obj_h) + base_h_border

            obj_box = Box(x_loc + obj_w // 2, y_loc + obj_h // 2, obj_w, obj_h)

            overlaps_other_obj = False
            for k in range(j - 1, -1, -1):
                other_box = Box(*yolo_boxes[k])
                if other_box.iomina(obj_box) > 0.25:
                    overlaps_other_obj = True
                    break
            if overlaps_other_obj:
                continue

            anchor_boxes = [Box(0, 0, *a) for a in self.yolo_network_info.anchors]

            # Get box at origin
            img_box_o = obj_box.offset(obj_box.x, obj_box.y)
            max_iou_idx = np.argmax([img_box_o.iou(a) for a in anchor_boxes])
            layer_idx = [i for i, indices in enumerate(self.yolo_network_info.masks) if max_iou_idx in indices][0]
            anchor_index = np.argmax(self.yolo_network_info.masks[layer_idx] == max_iou_idx)
            size = self.yolo_network_info.layer_sizes[layer_idx]

            # Determine the cell that is responsible for this prediction, i.e
            # the cell which contains the center of the box
            cell_x_idx = int(obj_box.x * size / net_width)
            cell_y_idx = int(obj_box.y * size / net_height)

            if yolo_truth[layer_idx][cell_y_idx, cell_x_idx, anchor_index, 4] == 1:
                continue

            transparency_mask = np.expand_dims(transformed_img[..., 3] != 0, -1)
            result_img[y_loc:y_loc + obj_h, x_loc: x_loc + obj_w] = \
                result_img[y_loc:y_loc + obj_h, x_loc: x_loc + obj_w] * (1 - transparency_mask) + \
                transformed_img[..., :3] * transparency_mask

            yolo_truth[layer_idx][cell_y_idx, cell_x_idx, anchor_index, :4] = obj_box.as_list()
            yolo_truth[layer_idx][cell_y_idx, cell_x_idx, anchor_index, 4] = 1
            yolo_truth[layer_idx][cell_y_idx, cell_x_idx, anchor_index, 5 + class_id] = 1
            yolo_boxes[j] = obj_box.as_list()
            j += 1

        # Random gamma correction on entire image
        # border_w = (self.yolo_network_info.input_size[0] - base_w) // 2
        # border_h = (self.yolo_network_info.input_size[1] - base_h) // 2
        # gamma = np.random.uniform(1/3, 1.5)
        # lut = ((np.arange(256) / 255) ** (1 / gamma) * 255).astype(np.uint8)
        # # Don't want to do gamma correction on the alpha channel
        # base_img_noalpha = (result_img[border_h:base_h + border_h, border_w:base_w + border_w, :3] * 255)\
        #     .astype(np.uint8)
        # result_img[border_h:base_h + border_h, border_w:base_w + border_w, :3]\
        #     = (cv2.LUT(base_img_noalpha, lut) / 255.0).astype(np.float32)

        return result_img, yolo_truth, yolo_boxes

    # Number of batches
    def __len__(self):
        # Number of batches
        return len(self.base_file_names) // self.batch_size

    # Return a batch of data
    def __getitem__(self, index):
        # Index into permuted indices to get permutation of images
        batch_file_names = [self.base_file_names[i] for i in
                             self.base_data_indices[index * self.batch_size: (index + 1) * self.batch_size]]

        # Read all base images and convert them to RGB, then letterbox them, which takes care of scaling
        base_images = [letterbox(self.yolo_network_info, cv2.cvtColor(cv2.imread(os.path.join(self.base_data_path, f)),
                                                                      cv2.COLOR_BGR2RGB)) for f in batch_file_names]

        data = [None] * len(base_images)
        for i, (base_img, new_w, new_h) in enumerate(base_images):
            if np.random.randint(0, 10) % 10 == 0:
                truths = [np.zeros((s, s, self.yolo_network_info.boxes_per_cell,
                                    4 + 1 + self.yolo_network_info.classes), dtype=np.float32)
                          for s in self.yolo_network_info.layer_sizes]
                boxes = np.zeros((self.yolo_network_info.max_boxes_per_img, 4), dtype=np.float32)
                data[i] = [base_img, truths, boxes]
            else:
                data[i] = self.insert_object_into_image(base_img, new_w, new_h)
                # if not self.is_val:
                # cv2.imshow("aug", (cv2.cvtColor(data[i][0], cv2.COLOR_RGBA2BGRA) * 255).astype(np.uint8))
                # cv2.waitKey(0)

        # Images
        images = np.stack([d[0] for d in data])
        yolo_truth = [np.stack([d[1][layer_idx] for d in data]) for layer_idx in
                      range(self.yolo_network_info.num_yolo_layers)]
        yolo_boxes = np.stack([d[2] for d in data])
        # print(index)
        return [images, yolo_truth[0], yolo_truth[1], yolo_boxes], \
               [np.zeros((images.shape[0], 4)), np.zeros((images.shape[0], 4))]

    def on_epoch_end(self):
        self.base_data_indices = np.random.permutation(len(self.base_file_names))
