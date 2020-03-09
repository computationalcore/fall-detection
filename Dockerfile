FROM ubuntu:16.04

ADD fall_detection.py /app/
ADD models /app/models/
ADD example /app/example/
ADD l_openvino_toolkit* /openvino/

WORKDIR /app

ARG INSTALL_DIR=/opt/intel/openvino

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        sudo 

# Installing OpenVINO dependencies
RUN cd /openvino/ && \
    ./install_openvino_dependencies.sh

RUN pip3 install numpy exitstatus

## Installing OpenVINO itself
RUN cd /openvino/ && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Model Optimizer
RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites && \
    ./install_prerequisites.sh
    
# clean up 
RUN apt autoremove -y && \
    rm -rf /openvino /var/lib/apt/lists/*

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"

RUN echo "source $INSTALL_DIR/bin/setupvars.sh" >> /root/.bashrc

CMD ["/bin/bash"]