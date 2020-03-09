# Fall Detection

[![](https://travis-ci.org/computationalcore/fall-detection.svg?branch=master)](https://travis-ci.org/computationalcore/fall-detection)

This project consists on showcasing the advantages of the Intel’s OpenVINO toolkit for inference in detecting people falling in an edge application.

This app perform single person fall detection using OpenVINO's [human-pose-estimation-0001](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html).

To detect falls, the app uses the coordinates of the head(nose, eyes and ears), neck and shoulders positions in a frame-by-frame comparison to determine
if the person is falling.

It works with a video file input or webcam stream.

[![](http://img.youtube.com/vi/C_S4oePpTZ8/0.jpg)](https://www.youtube.com/watch?v=C_S4oePpTZ8 "Fall Detection")

[Watch Video](https://www.youtube.com/watch?v=C_S4oePpTZ8)

## Prerequisites

To run the application in this tutorial, the OpenVINO™ toolkit and its dependencies must already be installed and verified using the included demos. 

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

The latest OpenVINO™ toolkit installed and verified. Supported versions +2018 R4.0. (Lastest version supported 2019 R1.0.1)

Git(git) for downloading from the GitHub repository.

BOOST library. To install on Ubuntu, run:

apt-get install libboost-dev
apt-get install libboost-log-dev

### Checks
By now you should have completed the Linux installation guide for the OpenVINO™ toolkit, however before continuing, please ensure:

That after installing the OpenVINO™ toolkit you have run the supplied demo samples

If you have and intend to use a GPU: You have installed and tested the GPU drivers

If you have and intend to use a USB camera: You have connected and tested the USB camera

If you have and intend to use a Myriad: You have connected and tested the USB Intel® Movidius™ Neural Compute Stick

That your development platform is connected to a network and has Internet access. To download all the files for this tutorial, you will need to access GitHub on the Internet.

## Build

- Clone the repository at desired location:

`git clone https://github.com/computationalcore/fall-detection`

- The first step is to configure the build environment for the OpenCV toolkit by sourcing the "setupvars.sh" script.

`source  /opt/intel/openvino/bin/setupvars.sh`

- For older versions than 2019 R1, OpenVINO was installed in a different dir, run this instead:

`source  /opt/intel/computer_vision_sdk/bin/setupvars.sh`

- Change to the top git repository:

`cd fall-detection`

## Run

### Detect falls on a video file (in this case example/demo.mp4, but it can be any other)
`python fall_detection.py -i example/demo.mp4`

### Detect falls on webcam
`python fall_detection.py -i cam`

## Limitations

- Works only with single person, it may be add in the future, but currently
multiples person in a scene will confuse the detector.

## Authors
Vin Busquet
* [https://github.com/computationalcore](https://github.com/computationalcore)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Changelog

For details, check out [CHANGELOG.md](CHANGELOG.md).