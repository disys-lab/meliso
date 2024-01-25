FROM ubuntu:22.04

RUN apt-get -y update && apt-get -y upgrade

RUN yes | apt-get install nano \
                    npm \
                    build-essential=12.9ubuntu3 \
                    libboost-all-dev=1.74.0.3ubuntu7 \
                    pkg-config=0.29.2-1ubuntu3 \
                    cmake=3.22.1-1ubuntu1.22.04.1 \
                    git=1:2.34.1-1ubuntu1.9 \
		    unzip

RUN apt-get -y install python3 python3-pip python3-numpy

RUN apt-get -y install openmpi-bin openmpi-doc libopenmpi-dev

RUN pip install Cython mpi4py

ADD . /meliso

WORKDIR "/meliso"

RUN PYTHONPATH=$PYTHONPATH:./build LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/ make all

ENV PYTHONPATH=$PYTHONPATH:./build

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/

CMD tail -f /dev/null
