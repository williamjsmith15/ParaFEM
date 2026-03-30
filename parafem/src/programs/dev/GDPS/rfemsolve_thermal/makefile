# MAKEFILE src/programs/dev/GDPS/rfemsolve_thermal/makefile
# -----------------------------------------------------------------------
# Compiles rfemsolve_thermal as RELEASE or DEBUG
# Links with RELEASE STATIC library libParaFEM_mpi.$(VERSION).a
#    or with DEBUG STATIC library libParaFEM_mpi_D.$(VERSION).a
# -----------------------------------------------------------------------

include ../../../../../build/mk_defs.inc
include ../../../../../build/$(MACHINE).inc

MODULE	= ../../../../../include/mpi
LIB	= ../../../../../lib
.SUFFIXES: .o .f90

all:	release

release:
	@echo
	@echo "Building rfemsolve_thermal RELEASE"
	$(MAKE) rfemsolve_thermal \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_XX_MPI_REL_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi.$(VERSION) $(DEFAULT_XX_MPI_REL_LDFLAGS)"
	@echo "Done rfemsolve_thermal RELEASE"

debug:
	@echo
	@echo "Building rfemsolve_thermal DEBUG"
	$(MAKE) rfemsolve_thermal \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_XX_MPI_DBG_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi_D.$(VERSION) $(DEFAULT_XX_MPI_DBG_LDFLAGS)"
	@echo "Done rfemsolve_thermal DEBUG"

rfemsolve_thermal: rfemsolve_thermal.o
	$(FC) rfemsolve_thermal.o -o rfemsolve_thermal $(LDFLAGS)

rfemsolve_thermal.o: rfemsolve_thermal.f90
	$(FC) rfemsolve_thermal.f90 $(FFLAGS)

clean:
	rm -f *.o

execlean:
	rm -f rfemsolve_thermal ../../../../../bin/rfemsolve_thermal

relink: rfemsolve_thermal

install:
	cp rfemsolve_thermal ../../../../../bin
