import sys
import os
import cv2
import time
import logging as log
import numpy as np

from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin

JOINT_COLORS = [
    (0, 0, 255),
    (0, 0, 128), (255, 255, 255), (0, 255, 0), (0, 0, 255),
    (192, 192, 192), (128, 0, 255), (0, 128, 128), (255, 255, 255),
    (128, 128, 0), (128, 128, 128), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 128)
]
JOINT_COLOR = (0, 255, 255)
POSE_PAIRS = [
    [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]
]


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)    

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
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
    
    out_filename = os.path.splitext(input_stream)[0]
    out = cv2.VideoWriter(out_filename + '_output.mp4', 0x00000021, fps, (width, height))
    
    log.info('Height %s' % height)
    log.info('font_scale %s' % font_scale)

    cur_request_id = 0
    next_request_id = 1

    # Detection variablez
    head_avg_position_previous = [0,0]
    previous_head_avg_position = 0
    previous_head_detection_frame = 0    
    fall_detected = False
    last_fall_detected_frame = 0
    # Fall Detection threshold speed is depedent of image height
    fall_threshold = 0.04 * height
    framerate_threshold = round(fps/5.0)
    fall_detected_text_position = (20,round(0.15*height))
    
   
        
        # Increment frame number
        frame_number += 1

        key = cv2.waitKey(1)
        if key == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)

