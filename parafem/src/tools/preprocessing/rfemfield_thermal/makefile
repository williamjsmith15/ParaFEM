# MAKEFILE src/tools/preprocessing/rfemfield_thermal/makefile
# -----------------------------------------------------------------------
# Compiles rfemfield_thermal executable as RELEASE or DEBUG
# Links with RELEASE STATIC library libParaFEM_mpi.$(VERSION).a
#    or with DEBUG STATIC library libParaFEM_mpi_D.$(VERSION).a
# Requires: gaf77 library (provides sim3de)
# -----------------------------------------------------------------------

include ../../../../build/mk_defs.inc
include ../../../../build/$(MACHINE).inc

MODULE	= ../../../../include/mpi
LIB	= ../../../../lib

.SUFFIXES: .o .f90

all:	release

release:
	@echo
	@echo "Building rfemfield_thermal RELEASE"
	$(MAKE) rfemfield_thermal \
	FFLAGS="-c -I$(MODULE) $(RFEMFIELD_MPI_REL_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi.$(VERSION) $(RFEMFIELD_MPI_REL_LDFLAGS) \
	-lgaf77"
	@echo "Done rfemfield_thermal RELEASE"

debug:
	@echo
	@echo "Building rfemfield_thermal DEBUG"
	$(MAKE) rfemfield_thermal \
	FFLAGS="-c -I$(MODULE) $(RFEMFIELD_MPI_DBG_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi_D.$(VERSION) $(RFEMFIELD_MPI_DBG_LDFLAGS) \
	         -lgaf77"
	@echo "Done rfemfield_thermal DEBUG"

rfemfield_thermal: rfemfield_thermal.o
	$(FC) rfemfield_thermal.o -o rfemfield_thermal $(LDFLAGS)

rfemfield_thermal.o: rfemfield_thermal.f90
	$(FC) rfemfield_thermal.f90 $(FFLAGS)

clean:
	rm -f *.o

execlean:
	rm -f rfemfield_thermal ../../../../bin/rfemfield_thermal

relink: rfemfield_thermal

install:
	$(INSTALL_CMD) rfemfield_thermal ../../../../bin
