# Use Caffe2 image as parent image
FROM caffe2:cuda8-cudnn6-all-options

#RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /cocoapi/PythonAPI
RUN make install

# Clone the Detectron repository
#RUN git clone https://github.com/facebookresearch/detectron /detectron
RUN git clone https://github.com/lg12170226/Detectron-restful /detectron


COPY retinanet.yaml /detectron/restful/model/
COPY model.pkl /detectron/restful/model/

# Set up Python modules
WORKDIR /detectron/lib
RUN make

# Build custom ops
RUN make ops

# Go to Detectron root
WORKDIR /detectron

# install flask
RUN pip install --upgrade pip
RUN pip install Flask

# install vim
RUN mv /var/lib/apt/lists /var/lib/apt/lists.bad
RUN apt-get update
RUN apt-get install vim

# start restful service
WORKDIR /detectron/restful
CMD python app.py
EXPOSE 8080
