# MAKEFILE src/programs/dev/GDPS/gdps_thermal_transient/makefile
# -----------------------------------------------------------------------
# Compiles gdps_thermal_transient as RELEASE or DEBUG
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
	@echo "Building gdps_thermal_transient RELEASE"
	$(MAKE) gdps_thermal_transient \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_XX_MPI_REL_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi.$(VERSION) $(DEFAULT_XX_MPI_REL_LDFLAGS)"
	@echo "Done gdps_thermal_transient RELEASE"

debug:
	@echo
	@echo "Building gdps_thermal_transient DEBUG"
	$(MAKE) gdps_thermal_transient \
	FFLAGS="-c -I$(MODULE) $(DEFAULT_XX_MPI_DBG_FFLAGS)" \
	LDFLAGS="-L$(LIB) -lParaFEM_mpi_D.$(VERSION) $(DEFAULT_XX_MPI_DBG_LDFLAGS)"
	@echo "Done gdps_thermal_transient DEBUG"

gdps_thermal_transient: gdps_thermal_transient.o
	$(FC) gdps_thermal_transient.o -o gdps_thermal_transient $(LDFLAGS)

gdps_thermal_transient.o: gdps_thermal_transient.f90
	$(FC) gdps_thermal_transient.f90 $(FFLAGS)

clean:
	rm -f *.o

execlean:
	rm -f gdps_thermal_transient ../../../../../bin/gdps_thermal_transient

relink: gdps_thermal_transient

install:
	cp gdps_thermal_transient ../../../../../bin
