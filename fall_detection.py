import argparse
import logging as log
import os
import sys

import cv2
from exitstatus import ExitStatus

from openvino.inference_engine import IENetwork, IEPlugin

JOINT_COLORS = [
    (0, 0, 255),
    (0, 0, 128), (255, 255, 255), (0, 255, 0), (0, 0, 255),
    (192, 192, 192), (128, 0, 255), (0, 128, 128), (255, 255, 255),
    (128, 128, 0), (128, 128, 128), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 128)
]

"""
Pose Data Points

Nose 0, Neck 1, Right Shoulder 2, Right Elbow 3, Right Wrist 4,
Left Shoulder 5, Left Elbow 6, Left Wrist 7, Right Hip 8,
Right Knee 9, Right Ankle 10, Left Hip 11, Left Knee 12,
LAnkle 13, Right Eye 14, Left Eye 15, Right Ear 16,
Left Ear 17, Background 18
"""
POSE_POINTS_NUMBER = 18
POSE_PAIRS = [
    [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]
]


def parse_args(parse_this=None) -> argparse.Namespace:
    """Parse user command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect a person falling from a webcam or a video file"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="""
            Path to video file or image. 'cam' for capturing
            video stream from internal camera.
        """,
        required=True,
        type=str
    )
    parser.add_argument(
        "-mp",
        "--model_precision",
        help="""The precision of the human pose model.
            Default is 32-bit integer.
        """,
        choices=('FP16', 'FP32'),
        default='FP32'
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="""MKLDNN (CPU)-targeted custom layers.Absolute path to a shared
            library with the kernels impl.
        """,
        type=str,
        default=None
    )
    parser.add_argument(
        "-pp",
        "--plugin_dir",
        help="Path to a plugin folder",
        type=str,
        default=None
    )
    parser.add_argument(
        "-d",
        "--device",
        help="""Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD
            is acceptable. Demo will look for a suitable plugin for device
            specified (CPU by default)
        """,
        default="CPU",
        type=str
    )
    return parser.parse_args(parse_this)


def main():
    """Accept arguments and perform the inference on entry"""
    # Setup log config
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout
    )

    # Parse args
    args = parse_args()
    log.info("Start Fall Detection")

    # Plugin initialization for specified device
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    # Load model
    model_xml = os.path.join(
        os.getcwd(),
        "models",
        "human-pose-estimation-0001",
        args.model_precision,
        "human-pose-estimation-0001.xml"
    )
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [
            l for l in net.layers.keys() if l not in supported_layers
        ]
        if len(not_supported_layers) != 0:
            log.error(
                """Following layers are not supported by the plugin
                for specified device {}:\n {}""".
                format(plugin.device, ', '.join(not_supported_layers))
            )
            log.error(
                """Please try to specify cpu extensions library
                    path in demo's command line parameters using -l
                    or --cpu_extension command line argument
                """
            )
            sys.exit(1)
    input_blob = next(iter(net.inputs))

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)

    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    font_scale = round(height/360)
    font_thickness = round(3*font_scale)

    cur_request_id = 0
    next_request_id = 1

    # Fall Detection variables
    previous_head_avg_position = 0
    previous_head_detection_frame = 0
    last_fall_detected_frame = 0
    # Fall Detection threshold speed is depedent of the frame height
    fall_threshold = 0.04 * height
    framerate_threshold = round(fps/5.0)
    fall_detected_text_position = (20, round(0.15*height))

    ret, frame = cap.read()
    frame_number = 0

    out_file = None
    if args.input != 'cam':
        out_filename = os.path.splitext(input_stream)[0] + '_output.mp4'
        out_file = cv2.VideoWriter(
            out_filename,
            0x00000021,
            fps,
            (width, height)
        )
        log.info("Evaluating video file stream...")
    else:
        log.info("Evaluating webcam stream...")

    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break

        # Pre-process inputs
        in_frame = cv2.resize(next_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        # Inference
        exec_net.start_async(
            request_id=next_request_id,
            inputs={input_blob: in_frame}
        )
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs
            kp_heatmaps = res['Mconv7_stage2_L2']

            threshold = 0.5
            points = []
            head_elements_y_pos = []

            for i in range(POSE_POINTS_NUMBER):
                # confidence map of corresponding body's part.
                probMap = kp_heatmaps[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                x = frame.shape[1] / probMap.shape[1] * point[0]
                y = frame.shape[0] / probMap.shape[0] * point[1]

                # Add point if the probability is greater than the threshold
                if prob > threshold:
                    point = (int(x), int(y))

                    # If point is a component of the head (including neck and
                    # sholders) append to the header elemnts
                    if (
                        i == 0 or
                        i == 1 or
                        i == 2 or
                        i == 5 or
                        i == 14 or
                        i == 15 or
                        i == 16 or
                        i == 17
                    ):
                        head_elements_y_pos.append(point[1])

                    points.append(point)
                else:
                    points.append(None)

            # Draw Skeleton
            for num, pair in enumerate(POSE_PAIRS):
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(
                        frame,
                        points[partA],
                        points[partB],
                        JOINT_COLORS[num],
                        3
                    )

            # Calculate head average position from its components
            if(len(head_elements_y_pos) > 0):
                head_avg_position = sum(head_elements_y_pos)
                head_avg_position /= len(head_elements_y_pos)
                # log.info(head_avg_position)

                # Compare previous head position
                # to detect if falling
                if (
                    previous_head_detection_frame and
                    (head_avg_position - previous_head_avg_position) >
                        fall_threshold and
                    (frame_number - previous_head_detection_frame) <
                        framerate_threshold
                ):
                    # print("Fall detected.")
                    last_fall_detected_frame = frame_number

                previous_head_avg_position = head_avg_position
                previous_head_detection_frame = frame_number

            # Draw Fall Detection Text if last fall event
            # ocurred max 2 seconds ago
            if (
                last_fall_detected_frame and
                (frame_number - last_fall_detected_frame) <= 2*fps
            ):
                cv2.putText(
                    frame,
                    "FALL DETECTED!",
                    fall_detected_text_position,
                    cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    (0, 0, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        # If webcam mode
        if out_file:
            out_file.write(frame)
        else:
            cv2.imshow("Detection Results", frame)

        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

        # Increment frame number
        frame_number += 1

        key = cv2.waitKey(1)
        if key == 27:
            break

    if out_file:
        # Release the out writer, capture, and destroy any OpenCV windows
        out_file.release()
        out_filename = os.path.splitext(input_stream)[0] + '_output.mp4'
        log.info("Finished. %s saved." % (out_filename))
    else:
        log.info("Finished.")
        cv2.destroyAllWindows()

    return ExitStatus.success

if __name__ == '__main__':
    sys.exit(main())
