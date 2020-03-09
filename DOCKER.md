# Build and Run the Docker image

The project provides a Dockerfile with a base environment to run your inference models with OpenVINO™.

## Prerequisites

In order to run this container you'll need to [install docker](https://docs.docker.com/install/).

## Building the Docker Image

If you haven't done it yet, the first step is to clone this repository

``` bash
git clone https://github.com/computationalcore/fall-detection
```

``` bash
cd fall-detection
```

### Download Intel® OpenVINO™ Toolkit

Next download the OpenVINO(tm) toolkit.

You can register and download it from the following link (Linux version):
[https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)

Or use wget to get the package directly (Latest version is 2020 1 by the time writing this guide)

``` bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz
```

### Extract the file in the project root folder

``` bash
tar -xf l_openvino_toolkit*
```

### Build the image

``` bash
docker build -t openvino-fall-detection .
```

## Using the image

### Run the container

You can directly run a container based on this image or use this image across other images.

To run a container based on this image and log inside the container:

``` bash
docker run -ti openvino-fall-detection /bin/bash
```

Inside the container, clone the project you can run the inference with the example video

``` bash
cd /app
python3 fall_detection.py -i example/demo.mp4
```

You can check more details about the options available in the project's [README file](README.md).
