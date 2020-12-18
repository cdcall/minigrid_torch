ARG CUDA="10.2"
ARG CUDNN="7"
ARG UBUNTU="18.04"
ARG GPU=TRUE
ARG VERSION="devel"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${VERSION}-ubuntu${UBUNTU}

ENV DEBIAN_FRONTEND noninteractive

# Install python 3.7, pip and system utilities
RUN apt-get update && apt-get install -y \
    apt-utils \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3-opencv \
    ca-certificates \
    wget \
    sudo \
    git \
    tmux \
    openssh-server \
    nano \
    lsof

RUN apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN chmod 1777 /tmp

# alias python
RUN ln -sv /usr/bin/python3.7 /usr/bin/python

# Update pip and setuptools (to avoid version conflict warnings)
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --upgrade setuptools

ENV PATH="/root/.local/bin:${PATH}"

# CHANGE username as needed
ARG username=rl_sandbox
RUN useradd -u 1000 -g sudo --create-home ${username}
WORKDIR /home/${username}
ARG HOME=/home/${username}

# install rl_sandbox components
COPY . ${HOME}/
# stop Jupyter whining
ENV PATH="${HOME}/.local/bin:${PATH}"
RUN pip install -e ${HOME}/.

RUN  chmod -R 777 ${HOME}
# RUN  chown -R ${username} ${HOME}

RUN ssh-keygen -b 2048 -t rsa -f /root/.ssh/id_rsa -q -N ""
RUN mv /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN service ssh start

# open up ssh, jupyter and tensorboard ports
EXPOSE 22/tcp
# EXPOSE 6006/tcp  to avoid port conflicts, using 6007
EXPOSE 6007/tcp
# EXPOSE 8888/tcp  to avoid port conflicts, using 8889
EXPOSE 8889/tcp

# deal with jupyter (can run as root -- not recommended -- or as rl_sandbox user)
RUN mkdir ${HOME}/.jupyter
COPY jupyter_notebook_config.txt ${HOME}/.jupyter/jupyter_notebook_config.py
RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.txt /root/.jupyter/jupyter_notebook_config.py

# ENV TERM xterm
# ENV HOME=/home/${username}
# USER ${username}

ENTRYPOINT ["bash"]



