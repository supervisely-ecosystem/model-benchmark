FROM supervisely/base-py-sdk:6.73.114
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install ipykernel
RUN pip3 install nbformat==4.2.0
RUN pip3 install executing==1.2.0

RUN pip3 install image-diversity
RUN pip3 install kaleido==0.2.1
RUN pip3 install cupy-cuda111

RUN pip3 install zstd
RUN pip3 install pycocotools
RUN pip3 install scikit-learn

LABEL "role"="development"