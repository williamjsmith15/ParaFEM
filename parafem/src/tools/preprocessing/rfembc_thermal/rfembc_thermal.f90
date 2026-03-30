PROGRAM rfembc_thermal

!/****h* tools/preprocessing/rfembc_thermal
!*  NAME
!*    PROGRAM rfembc_thermal
!*  FUNCTION
!*    Loads an unstructured mesh and finds zmin and zmax face nodes.
!*    Generates a .fix file prescribing a concentration (or temperature)
!*    on both faces. zmin receives val_downstream, zmax receives val_upstream.
!*    Produces an updated .dat with correct nr=0 and fixed_freedoms count.
!*    Intended for use with rfemsolve_thermal (nodof=1).
!*  USAGE
!*    rfembc_thermal <job_name> <val_upstream> <val_downstream>
!*  INPUTS
!*    <job_name>.dat - Control file (rfemsolve format from rfemcube)
!*    <job_name>.d   - Mesh geometry file
!*  OUTPUTS
!*    <job_name>.fix - Fixed freedoms (both upstream zmax and downstream zmin)
!*    <job_name>.dat - Updated with fixed_freedoms count (nr=0, no .bnd needed)
!*    <job_name>.res - Summary (node counts, extents)
!*  AUTHORS
!*    Adapted from rfembc by Louise M. Lever, Lee Margetts
!*  COPYRIGHT
!*    (c) University of Manchester 2012, 2026
!****
!*/

  USE PRECISION

  IMPLICIT NONE

!------------------------------------------------------------------------------
! 1. Declare variables used in the main program
!------------------------------------------------------------------------------

  INTEGER                :: argc,iargc
  INTEGER                :: ndim,i
  CHARACTER(LEN=50)      :: job_name,res_name,in_dat_name,out_dat_name,d_name
  CHARACTER(LEN=50)      :: fix_name
  CHARACTER(LEN=15)      :: upstream_arg,downstream_arg
  CHARACTER(LEN=15)      :: element
  INTEGER                :: nels,nn,nr,nod,nip,loaded_nodes
  INTEGER                :: limit,mesh,fixed_freedoms,partition,np_types
  REAL(iwp)              :: tol,mises
  REAL(iwp)              :: val_upstream,val_downstream
  INTEGER                :: idummy
  INTEGER                :: zord=3
  REAL(iwp)              :: zmin_extent,zmax_extent
  INTEGER                :: num_fix=0,j

!------------------------------------------------------------------------------
! 2. Declare dynamic arrays
!------------------------------------------------------------------------------

  REAL(iwp), ALLOCATABLE :: m_coord(:,:)
  INTEGER,   ALLOCATABLE :: fix_node(:),fix_order(:)
  REAL(iwp), ALLOCATABLE :: fix_val(:)

!------------------------------------------------------------------------------
! 3. Read job_name and BC values from the command line
!------------------------------------------------------------------------------

  argc = iargc()
  IF (argc /= 3) THEN
    PRINT*
    PRINT*, "Usage:  rfembc_thermal <job_name> <val_upstream> <val_downstream>"
    PRINT*
    PRINT*, "        program expects <job_name>.d and <job_name>.dat"
    PRINT*, "        and outputs <job_name>.fix and <job_name>.dat (updated)"
    PRINT*
    PRINT*, "        zmax nodes receive val_upstream"
    PRINT*, "        zmin nodes receive val_downstream"
    PRINT*
    STOP
  END IF
  CALL GETARG(1, job_name)
  CALL GETARG(2, upstream_arg)
  CALL GETARG(3, downstream_arg)
  READ(upstream_arg,*)   val_upstream
  READ(downstream_arg,*) val_downstream

!------------------------------------------------------------------------------
! 4. Open result file
!------------------------------------------------------------------------------

  res_name = job_name(1:INDEX(job_name," ")-1) // ".res"
  OPEN(11,FILE=res_name,STATUS='REPLACE',ACTION='WRITE')

!------------------------------------------------------------------------------
! 5. Read model dat and coords; determine z extents
!------------------------------------------------------------------------------

  IF (INDEX(job_name,".dat") /= 0) THEN
     job_name = job_name(1:INDEX(job_name,".dat")-1)
  END IF
  in_dat_name  = job_name(1:INDEX(job_name," ")-1) // ".dat"
  out_dat_name = job_name(1:INDEX(job_name," ")-1) // ".dat"
  d_name       = job_name(1:INDEX(job_name," ")-1) // ".d"

  OPEN (14, file=in_dat_name, status='old', action='read')
  READ(14,*) element,mesh,partition,np_types,   &
       nels,nn,nr,nip,nod,loaded_nodes,         &
       fixed_freedoms,tol,limit,mises
  CLOSE(14)

  ndim = 3
  ALLOCATE(m_coord(ndim,nn))

  OPEN (15, file=d_name, status='old', action='read')
  READ(15,*)   ! *THREE_DIMENSIONAL
  READ(15,*)   ! *NODES

  READ(15,*) idummy,m_coord(:,1)
  zmin_extent = m_coord(zord,1)
  zmax_extent = m_coord(zord,1)
  DO i = 2,nn
     READ(15,*) idummy,m_coord(:,i)
     IF ( m_coord(zord,i) < zmin_extent ) zmin_extent = m_coord(zord,i)
     IF ( m_coord(zord,i) > zmax_extent ) zmax_extent = m_coord(zord,i)
  END DO

  CLOSE(15)

  WRITE(11, '(A,I12)') "Number of model nodes = ", nn
  WRITE(11, '(A,I12)') "Number of model elems = ", nels
  WRITE(11, '(A,I12)') "Number of model nods  = ", nod
  WRITE(11, '(A,2F14.6)') "ZMIN/ZMAX of model: ", zmin_extent, zmax_extent
  WRITE(11, '(A,E14.6)') "Upstream   value (zmax): ", val_upstream
  WRITE(11, '(A,E14.6)') "Downstream value (zmin): ", val_downstream

!------------------------------------------------------------------------------
! 6. Collect fixed nodes and sort in ascending node number order.
!    find_no2 (ParaFEM library) requires nodes to be in ascending order.
!    Write .fix file: upstream face (zmax) and downstream face (zmin).
!    nodof=1, DOF sense is always 1.
!------------------------------------------------------------------------------

  DO i = 1,nn
     IF( m_coord(zord,i) == zmax_extent .OR. &
         m_coord(zord,i) == zmin_extent ) THEN
        num_fix = num_fix + 1
     END IF
  END DO

  ALLOCATE(fix_node(num_fix),fix_val(num_fix),fix_order(num_fix))

  j = 0
  DO i = 1,nn
     IF( m_coord(zord,i) == zmax_extent ) THEN
        j = j + 1
        fix_node(j) = i
        fix_val(j)  = val_upstream
     ELSE IF( m_coord(zord,i) == zmin_extent ) THEN
        j = j + 1
        fix_node(j) = i
        fix_val(j)  = val_downstream
     END IF
  END DO

  DO i = 1,num_fix
     fix_order(i) = i
  END DO
  CALL sort_nodes(fix_node,fix_order,num_fix)

  fix_name = job_name(1:INDEX(job_name," ")-1) // ".fix"
  OPEN(17,FILE=fix_name,STATUS='REPLACE',ACTION='WRITE')
  DO i = 1,num_fix
     WRITE(17,'(I12,A,E14.6)') fix_node(i), " 1", fix_val(fix_order(i))
  END DO
  CLOSE(17)

  DEALLOCATE(fix_node,fix_val,fix_order)

  WRITE(11, '(A,I12)') "Number of fixed freedoms = ", num_fix

  DEALLOCATE(m_coord)

!------------------------------------------------------------------------------
! 7. Rewrite .dat file with nr=0, fixed_freedoms=num_fix
!    nr=0 because all BCs are Dirichlet (penalty method via .fix); no .bnd needed
!------------------------------------------------------------------------------

  OPEN (14, file=out_dat_name, status='REPLACE', action='WRITE')
  WRITE(14,*) element
  WRITE(14,*) mesh
  WRITE(14,*) partition
  WRITE(14,*) np_types
  WRITE(14, '(7I12)') nels, nn, 0, nip, nod, loaded_nodes, num_fix
  WRITE(14, '(E14.6,I12,E14.6)') tol, limit, mises
  CLOSE(14)

!------------------------------------------------------------------------------
! 8. Close result file
!------------------------------------------------------------------------------

  CLOSE(11)

CONTAINS

  SUBROUTINE sort_nodes(arr, order, n)
    INTEGER, INTENT(INOUT) :: arr(:), order(:)
    INTEGER, INTENT(IN)    :: n
    INTEGER                :: i, j, tmp_a, tmp_o
    DO i = 2, n
       tmp_a = arr(i); tmp_o = order(i); j = i - 1
       DO WHILE (j >= 1 .AND. arr(j) > tmp_a)
          arr(j+1) = arr(j); order(j+1) = order(j); j = j - 1
       END DO
       arr(j+1) = tmp_a; order(j+1) = tmp_o
    END DO
  END SUBROUTINE sort_nodes

END PROGRAM rfembc_thermal
