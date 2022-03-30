FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /root

# install utilities

RUN \
    DEBIAN_FRONTEND="noninteractive" apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y rsync byobu tmux vim nano htop wget curl git lm-sensors openssh-server && \
    mkdir .ssh
EXPOSE 22

# install conda
RUN \
    wget -O miniconda.sh "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
ENV PATH=/root/miniconda3/bin:${PATH}
RUN conda update -y conda && conda init

# setup env
WORKDIR /classy
COPY . .
RUN \
    bash -c "source ~/miniconda3/etc/profile.d/conda.sh && printf 'classy\n3.8\n1.10.2\n11.3\nN\n' | bash setup.sh"

# standard cmd
CMD [ "/bin/bash" ]
