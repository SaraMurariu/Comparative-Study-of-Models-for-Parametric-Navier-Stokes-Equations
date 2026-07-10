## Set Base Stage
FROM ubuntu:24.04 AS rbm-base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
apt-utils \
ssh \
software-properties-common \
apt-transport-https ca-certificates gnupg software-properties-common wget \
bash-completion

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
add-apt-repository -y ppa:deadsnakes/ppa && \
apt-get update

RUN apt update && \
apt install -y git

RUN apt-get install -y python3.13 && \
apt-get install -y python3.13-full && \
apt-get install -y python3.13-dev && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 100 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.13 100 && \
python3 -m ensurepip --upgrade && \
python3 -m pip install --upgrade pip


RUN pip3 install jupyter --no-cache-dir && \
pip3 install numpy --no-cache-dir && \
pip3 install scipy --no-cache-dir && \
pip3 install matplotlib --no-cache-dir && \
pip3 install vtk --no-cache-dir && \
pip3 install tqdm --no-cache-dir && \
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir && \
pip3 install pypolydim --no-cache-dir

WORKDIR /content
RUN git clone https://github.com/fvicini/RBM_ML_2026.git
WORKDIR /content/RBM_ML_2026

RUN echo "source /usr/share/bash-completion/completions/git;cd /content/RBM_ML_2026;git pull;" > /content/on_startup.sh
RUN chmod +x /content/on_startup.sh

WORKDIR /shared
CMD jupyter notebook --ip 0.0.0.0 --port=8080 --no-browser --allow-root
