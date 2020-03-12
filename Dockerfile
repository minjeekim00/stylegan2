# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

# default
#FROM tensorflow/tensorflow:1.15.0-gpu-py3

#RUN pip install scipy==1.3.3
#RUN pip install requests==2.22.0
#RUN pip install Pillow==6.2.1

# custom
FROM m40030811/stylegan2:latest

# for docker
RUN apt-get update
RUN apt-get install -y sudo vim
RUN addgroup --gid 1000001 gusers
RUN adduser --uid 1000018 --gid 1000001 --disabled-password --gecos '' minjee

RUN adduser minjee sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL'>> /etc/sudoers

USER minjee
