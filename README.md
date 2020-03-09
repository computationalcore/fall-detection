# Fall Detection

[![](https://travis-ci.org/computationalcore/fall-detection.svg?branch=master)](https://travis-ci.org/computationalcore/fall-detection)

This project consists on showcasing the advantages of the Intel’s OpenVINO toolkit for inference in detecting people falling in an edge application.

This app perform single person fall detection using OpenVINO's [human-pose-estimation-0001](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html) pre-trained model.

To detect falls, the app uses the coordinates of the head(nose, eyes and ears), neck and shoulders positions in a frame-by-frame comparison to determine
if the person is falling.

It works with a video file input or webcam stream.

[![](http://img.youtube.com/vi/C_S4oePpTZ8/0.jpg)](https://www.youtube.com/watch?v=C_S4oePpTZ8 "Fall Detection")

[Watch Video](https://www.youtube.com/watch?v=C_S4oePpTZ8)

## Prerequisites

To run the application in this tutorial, the OpenVINO™ toolkit and its dependencies must already be installed. 

Alternatively, you can create and build the docker image provided by this repository by following these [instructions](DOCKER.md).

Installation instructions may be found at: https://software.intel.com/en-us/articles/OpenVINO-Install-Linux

When needed, the following optional hardware can be used:

- USB camera - Standard USB Video Class (UVC) camera.

- Intel® Core™ CPU with integrated graphics.

- VPU - USB Intel® Movidius™ Neural Compute Stick and what is being referred to as "Myriad"

A summary of what is needed:

### Hardware
Target and development platforms meeting the requirements described in the "System Requirements" section of the OpenVINO™ toolkit documentation which may be found at: https://software.intel.com/en-us/openvino-toolkit

Note: While writing this tutorial, an Intel® i7-8550U with Intel® HD graphics 520 GPU was used as both the development and target platform.

Optional:

- Intel® Movidius™ Neural Compute Stick

- USB UVC camera

- Intel® Core™ CPU with integrated graphics

### Software
OpenVINO™ toolkit supported Linux operating system. This tutorial was run on 64-bit Ubuntu 16.04.1 LTS updated to kernel 4.15.0-43 following the OpenVINO™ toolkit installation instructions.

- The latest OpenVINO™ toolkit installed and verified. Supported versions +2018 R4.0. (Lastest version supported 2019 R1.0.1)

- Git(git) for downloading from the GitHub repository.

- BOOST library. 

To install on Ubuntu, run

``` bash
apt-get install libboost-dev libboost-log-dev
```

### Checks
By now you should have completed the Linux installation guide for the OpenVINO™ toolkit, however before continuing, please ensure:

- That after installing the OpenVINO™ toolkit you have run the supplied demo samples

- If you have and intend to use a GPU: You have installed and tested the GPU drivers

- If you have and intend to use a USB camera: You have connected and tested the USB camera

- If you have and intend to use a Myriad: You have connected and tested the USB Intel® Movidius™ Neural Compute Stick


## Build

- Clone the repository at desired location:

``` bash
git clone https://github.com/computationalcore/fall-detection
```

- The first step is to configure the build environment for the OpenCV toolkit by sourcing the "setupvars.sh" script.

``` bash
source  /opt/intel/openvino/bin/setupvars.sh
```

- For older versions than 2019 R1, OpenVINO was installed in a different dir, run this instead:

``` bash
source  /opt/intel/computer_vision_sdk/bin/setupvars.sh
```


- Change to the top git repository:

``` bash
cd fall-detection
```

- Install other project dependencies after run openvino env

``` bash
pip install -r requirements.txt
```

## Run

To check available options run

``` bash
$ python fall_detection.py -h
usage: fall_detection.py [-h] -i INPUT [-mp {FP16,FP32}] [-l CPU_EXTENSION] [-pp PLUGIN_DIR] [-d DEVICE]

Detect a person falling from a webcam or a video file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video stream from internal camera.
  -mp {FP16,FP32}, --model_precision {FP16,FP32}
                        The precision of the human pose model. Default is 32-bit integer.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.
  -pp PLUGIN_DIR, --plugin_dir PLUGIN_DIR
                        Path to a plugin folder
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo will look for a
                        suitable plugin for device specified (CPU by default)
```

### Detecting falls on a video file

(in this case I use example/demo.mp4, but it can be any other)

``` bash
python fall_detection.py -i example/demo.mp4
```

### Detecting falls on webcam
``` bash
python fall_detection.py -i cam
```

## Limitations

- It works only with a single person, in the future I can add multiple people support, but currently if there is
more than one person in the scene it will confuse the detector.
- The detector only takes into consideration the relative positions of head elements, neck and shoulders. In the future it can be improved
to consider other aspects of the human pose elements 

## Authors
Vin Busquet
* [https://github.com/computationalcore](https://github.com/computationalcore)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Changelog

For details, check out [CHANGELOG.md](CHANGELOG.md).