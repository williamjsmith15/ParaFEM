# MAKEFILE src/tools/preprocessing/rfembc_thermal/makefile
# -----------------------------------------------------------------------
# Compiles rfembc_thermal executable as RELEASE or DEBUG
# Links with RELEASE STATIC library libParaFEM_mpi.$(VERSION).a
#    or with DEBUG STATIC library libParaFEM_mpi_D.$(VERSION).a
# -----------------------------------------------------------------------

include ../../../../build/mk_defs.inc
include ../../../../build/$(MACHINE).inc

MODULE	= ../../../../include/mpi
LIB	= ../../../../lib

.SUFFIXES: .o .f90

all:	release

release:
	@echo
	@echo "Building rfembc_thermal RELEASE"
	$(MAKE) rfembc_thermal \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_PREPROC_MPI_REL_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi.$(VERSION) $(DEFAULT_PREPROC_MPI_REL_LDFLAGS)"
	@echo "Done rfembc_thermal RELEASE"

debug:
	@echo
	@echo "Building rfembc_thermal DEBUG"
	$(MAKE) rfembc_thermal \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_PREPROC_MPI_DBG_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi_D.$(VERSION) $(DEFAULT_PREPROC_MPI_DBG_LDFLAGS)"
	@echo "Done rfembc_thermal DEBUG"

rfembc_thermal: rfembc_thermal.o
	$(FC) rfembc_thermal.o -o rfembc_thermal $(LDFLAGS)

rfembc_thermal.o: rfembc_thermal.f90
	$(FC) rfembc_thermal.f90 $(FFLAGS)

clean:
	rm -f *.o

execlean:
	rm -f rfembc_thermal ../../../../bin/rfembc_thermal

relink: rfembc_thermal

install:
	$(INSTALL_CMD) rfembc_thermal ../../../../bin
