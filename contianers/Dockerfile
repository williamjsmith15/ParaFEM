FROM ubuntu:20.04

RUN apt-get update && \
	apt-get install -y  make \
			mpich \
			gfortran \
			gawk

# RUN apt install -y git-core

RUN mkdir /parafem

COPY ./parafem /parafem

WORKDIR /parafem

RUN ./make-parafem MACHINE=linuxdesktop

ENV PATH ="${PATH}":/parafem/bin
WORKDIR /home/ParaFEM

ENTRYPOINT /bin/bash
