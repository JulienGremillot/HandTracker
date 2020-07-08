import csv
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

# Can't make it work with outputs from the model downloaded from https://github.com/google/mediapipe/tree/master/mediapipe/models
# Using an older version of the model downloaded from https://github.com/metalwhale/hand_tracking/tree/master/models
HAND_LANDMARK_MODEL = "models/hand_landmark.tflite"

# When I tried the palm_detection.tflite model I got an error with custom operation : "Encountered unresolved custom op: Convolution2DTransposeBias.Node number 165 (Convolution2DTransposeBias)"
# After reading this thread : https://github.com/google/mediapipe/issues/35 I downloaded a model without custom op from https://github.com/metalwhale/hand_tracking
PALM_DETECTION_MODEL = "models/palm_detection_without_custom_op.tflite"

# SSD anchors from https://github.com/SYSU-RealTimeHandGesture-2020/papers/tree/4a287a66f2565c4ffa4f756cfee0b7043ec2b966/gg_handtracking/python-handtracking/data
SSD_ANCHORS = "resources/ssd_anchors.csv"

BOX_ENLARGE = 1.5
BOX_SHIFT = 0.2

# triangle target coordinates used to move the detected hand into the right position
TARGET_TRIANGLE = np.float32([
    [128, 128],
    [128, 0],
    [0, 128]
])
TARGET_BOX = np.float32([
    [0, 0, 1],
    [256, 0, 1],
    [256, 256, 1],
    [0, 256, 1],
])

# 90° rotation matrix used to create the alignment triangle
R90 = np.r_[[[0, 1], [-1, 0]]]

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default="cam", type=str,
                        help="Value can be 'cam' or a path to image or video file")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def preprocess_image(image, input_shape):
    """
    Resize the image in 256x256

    :return: img_pad, img_norm, pad
    """

    shape = np.r_[image.shape]
    pad = (shape.max() - shape[:2]).astype('uint32') // 2
    img_pad = np.pad(image, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), mode='constant')
    img_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
    img_norm = img_resized.astype('float32') / 255.0
    img_norm = img_norm.reshape(1, *img_norm.shape)
    return img_pad, img_norm, pad[::-1]


def get_triangle(kp0, kp2, dist=1):
    """get a triangle used to calculate Affine transformation matrix"""
    dir_v = kp2 - kp0
    dir_v /= np.linalg.norm(dir_v)
    dir_v_r = dir_v @ R90.T
    return np.float32([kp2, kp2 + dir_v * dist, kp2 + dir_v_r * dist])


def detect_hand(image, palm_detection, anchors, prob_threshold):
    assert -1 <= image.min() and image.max() <= 1, \
        "image should be in range [-1, 1]"
    assert image.shape == (1, 256, 256, 3), \
        "image shape must be (1, 256, 256, 3)"

    # predict hand location and 7 initial landmarks
    in_idx = palm_detection.get_input_details()[0]['index']
    palm_detection.set_tensor(in_idx, image)
    palm_detection.invoke()

    out_reg = palm_detection.get_tensor(palm_detection.get_output_details()[0]['index'])[0]
    out_clf = palm_detection.get_tensor(palm_detection.get_output_details()[1]['index'])[0, :, 0]

    # finding the best prediction
    detection_mask = (1 / (1 + np.exp(-out_clf))) > prob_threshold
    candidate_detect = out_reg[detection_mask]
    candidate_anchors = anchors[detection_mask]

    if candidate_detect.shape[0] == 0:
        return None, None

    # picking the widest suggestion while NMS is not implemented
    max_idx = np.argmax(candidate_detect[:, 3])

    # bounding box offsets, width and height
    dx, dy, w, h = candidate_detect[max_idx, :4]
    center_wo_offst = candidate_anchors[max_idx, :2] * 256

    # 7 initial keypoints
    keypoints = center_wo_offst + candidate_detect[max_idx, 4:].reshape(-1, 2)

    # now we need to move and rotate the detected hand for it to occupy a 256x256 square
    # line from wrist keypoint to middle finger keypoint should point straight up
    # TODO: replace triangle with the bbox directly
    source = get_triangle(keypoints[0], keypoints[2], max(w, h) * BOX_ENLARGE)
    source -= (keypoints[0] - keypoints[2]) * BOX_SHIFT
    return source, keypoints


def predict_joints(img_norm, hand_landmark):
    hand_landmark.set_tensor(hand_landmark.get_input_details()[0]['index'], img_norm.reshape(1, 256, 256, 3))
    hand_landmark.invoke()
    joints = hand_landmark.get_tensor(hand_landmark.get_output_details()[0]['index'])
    return joints.reshape(-1, 2)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    palm_detection = tf.lite.Interpreter(PALM_DETECTION_MODEL)
    palm_detection.allocate_tensors()
    hand_landmark = tf.lite.Interpreter(HAND_LANDMARK_MODEL)
    hand_landmark.allocate_tensors()

    # reading the SSD anchors
    with open(SSD_ANCHORS, "r") as csv_f:
        anchors = np.r_[
            [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
        ]

    # Handle the input
    is_single_image_mode = os.path.splitext(args.input)[1] in ['.jpg', '.png']

    if not is_single_image_mode:
        if args.input == "cam":
            print("Init webcam capture")
            cap = cv2.VideoCapture(0)
        else:
            print("Opening video file", args.input)
            cap = cv2.VideoCapture(args.input)

    ### Loop until stream is over ###
    while is_single_image_mode or cap.isOpened():

        if is_single_image_mode:
            print("Single image mode. Analyze ", args.input)
            frame = cv2.imread(args.input)
        else:
            ### Read from the video capture ###
            flag, frame = cap.read()
            if not flag:
                break
            key_pressed = cv2.waitKey(60)
            if args.input == "cam":
                frame = cv2.flip(frame, 1)

        img_pad, img_norm, pad = preprocess_image(frame, palm_detection.get_input_details()[0]['shape'])

        source, keypoints = detect_hand(img_norm, palm_detection, anchors, args.prob_threshold)
        if source is None:
            continue

        # calculating transformation from img_pad coords to img_landmark coords
        scale = max(frame.shape) / 256
        Mtr = cv2.getAffineTransform(source * scale, TARGET_TRIANGLE)

        img_pad = img_pad.astype('float32')
        img_pad /= 255.0
        img_landmark = cv2.warpAffine(img_pad, Mtr, (256, 256))

        joints = predict_joints(img_landmark, hand_landmark)

        # adding the [0,0,1] row to make the matrix square
        Mtr = np.pad(Mtr.T, ((0, 0), (0, 1)), constant_values=1, mode='constant').T
        Mtr[2, :2] = 0
        Minv = np.linalg.inv(Mtr)

        # projecting keypoints back into original image coordinate space
        joints_padded = np.pad(joints, ((0, 0), (0, 1)), constant_values=1, mode='constant')
        kp = (joints_padded @ Minv.T)[:, :2] - pad
        box = (TARGET_BOX @ Minv.T)[:, :2] - pad

        keyid = 0
        for keypoint in kp:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), radius=2, color=(255, 0, 0), thickness=3)
            if keyid not in [0, 5, 9, 13, 17]:
                cv2.line(frame, (int(keypoint[0]), int(keypoint[1])), (int(kp[keyid - 1][0]), int(kp[keyid - 1][1])),
                         color=(255, 0, 0), thickness=2)
            if keyid in [5, 9, 13, 17]:
                cv2.line(frame, (int(keypoint[0]), int(keypoint[1])), (int(kp[0][0]), int(kp[0][1])),
                         color=(255, 0, 0), thickness=2)
            keyid += 1
        pts = np.array(box, np.int32)
        cv2.polylines(frame, [pts], True, (0,255,255))

        # Write an output image if is in single_image_mode
        if is_single_image_mode:
            print("Write output file in 'single_image.png'")
            cv2.imwrite('single_image.png', frame)
        else:
            cv2.imshow('', frame)
            cv2.waitKey(1)

        # Break if single_image_mode or escape key pressed
        if is_single_image_mode or key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    if not is_single_image_mode:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()