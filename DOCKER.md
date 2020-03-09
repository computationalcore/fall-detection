# Build and Run Docker image

The project provides a Dockerfile with a base environment to run your inference models with OpenVINO™.

## Prerequisites

In order to run this container you'll need to [install docker](https://docs.docker.com/install/).

## Building the Docker Image

### Download Intel® OpenVINO™ Toolkit

The first thing you need is to download the OpenVINO(tm) toolkit.

You can register and download it from the following link (Linux version):
[https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)

Or use wget to get the package directly (Latest version is 2019 R3.1 by the time writing this guide)

``` bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
```

### Extract the file in the root folder

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

To run a container based on this image:

``` bash
docker run -ti openvino-fall-detection /bin/bash
```

Download the project

``` bash
docker exec openvino-fall-detection git clone https://github.com/computationalcore/fall-detection
```

To run the project with the example video inference

``` bash
docker exec openvino-fall-detection bash -c 'source /opt/intel/openvino/bin/setupvars.sh && cd /app/fall-detection && python3 fall_detection.py -i example/demo.mp4'
```

You can check more details about the options available in the project's [README file](README.md).

### Run the the container with X enabled (Linux)

Additionally, for running a sample application that displays an image, you need to share the host display to be accessed from guest Docker container.

The X server on the host should be enabled for remote connections:

``` bash
xhost +
```

The following flags needs to be added to the docker run command:

* --net=host
* --env="DISPLAY"
* --volume="$HOME/.Xauthority:/root/.Xauthority:rw"

To run the sample-app image with the display enabled:

``` bash
docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -ti sample-app /bin/bash
```

Finally disable the remote connections to the X server

``` bash
xhost -
```