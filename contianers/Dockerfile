FROM ubuntu:20.04 AS build

RUN apt-get update && \
    apt-get install -y make mpich gfortran gawk && \
    rm -rf /var/lib/apt/lists/*

COPY ./parafem /parafem
WORKDIR /parafem
RUN ./make-parafem MACHINE=linuxdesktop
RUN cd src/programs/dev/GDPS/gdps_thermal_transient && \
    make MACHINE=linuxdesktop && make install MACHINE=linuxdesktop

FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y mpich gawk && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /parafem/bin/ /usr/local/bin/
COPY --from=build /parafem/src/tools/preprocessing/inp2pf/inp2pf /usr/local/bin/
COPY --from=build /parafem/src/tools/preprocessing/inp2pf/inp2pf.awk /usr/local/bin/
COPY --from=build /parafem/src/tools/postprocessing/pf2ensi/pf2ensi /usr/local/bin/
COPY --from=build /parafem/src/tools/postprocessing/pf2ensi/pf2ensi.geo.awk /usr/local/bin/
COPY --from=build /parafem/src/tools/postprocessing/pf2ensi/pf2ensi.var.awk /usr/local/bin/

WORKDIR /work
