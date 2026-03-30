PROGRAM rfemsolve_thermal
!------------------------------------------------------------------------------
! 3D steady-state diffusion / thermal analysis with spatially varying
! material properties (per-element diffusivity or conductivity).
!
! Based on p123 (Laplace PCG solver) with instance-based file naming
! and per-element material reading from rfemfield_thermal output.
!
! Usage: rfemsolve_thermal <model_name> <instance-id>
!
! Inputs:
!   <model_name>-<instance-id>.d   - mesh (from rfemfield_thermal)
!   <model_name>-<instance-id>.dat - control (rfemsolve format, np_types=nels)
!   <model_name>-<instance-id>.mat - per-element diffusivity
!   <model_name>.bnd               - restrained nodes (shared across instances)
!   <model_name>.fix               - fixed freedoms (shared across instances)
!
! Outputs:
!   <model_name>-<instance-id>.res        - convergence summary
!   <model_name>-<instance-id>.ensi.NDPTL - scalar field for ParaView
!------------------------------------------------------------------------------

! USE mpi_wrapper  ! uncomment for serial compilation

  USE precision     ; USE global_variables ; USE mp_interface
  USE input         ; USE output           ; USE loading
  USE timing        ; USE maths            ; USE gather_scatter
  USE new_library

  IMPLICIT NONE

!------------------------------------------------------------------------------
! 1. Declare variables
!------------------------------------------------------------------------------

  INTEGER,PARAMETER     :: nodof=1,ndim=3,nprops=1
  INTEGER               :: loaded_freedoms,fixed_freedoms,iel,i,j,k
  INTEGER               :: iters,limit,nn,nr,nip,nod,nels,ndof,npes_pp
  INTEGER               :: node_end,node_start,nodes_pp
  INTEGER               :: argc,iargc,meshgen,partitioner,np_types
  INTEGER               :: fixed_freedoms_pp,fixed_freedoms_start
  INTEGER               :: loaded_freedoms_pp,loaded_freedoms_start
  INTEGER               :: nres,is,it
  REAL(iwp)             :: det,tol,up,alpha,beta,q=0.0_iwp
  REAL(iwp)             :: mises,kx,ky,kz
  REAL(iwp),PARAMETER   :: zero=0.0_iwp,penalty=1.e20_iwp
  CHARACTER(LEN=6)      :: ch
  CHARACTER(LEN=15)     :: element
  CHARACTER(LEN=50)     :: fname,job_in,instance_id,inst_in,base_in
  LOGICAL               :: converged=.false.

!------------------------------------------------------------------------------
! 2. Declare dynamic arrays
!------------------------------------------------------------------------------

  REAL(iwp),ALLOCATABLE :: points(:,:),weights(:),eld_pp(:,:)
  REAL(iwp),ALLOCATABLE :: jac(:,:),der(:,:),deriv(:,:)
  REAL(iwp),ALLOCATABLE :: col(:,:),row(:,:)
  REAL(iwp),ALLOCATABLE :: kcx(:,:),kcy(:,:),kcz(:,:)
  REAL(iwp),ALLOCATABLE :: diag_precon_pp(:),p_pp(:),r_pp(:),x_pp(:),xnew_pp(:)
  REAL(iwp),ALLOCATABLE :: u_pp(:),pmul_pp(:,:),utemp_pp(:,:),d_pp(:)
  REAL(iwp),ALLOCATABLE :: diag_precon_tmp(:,:),store_pp(:),val_f(:)
  REAL(iwp),ALLOCATABLE :: storkc_pp(:,:,:),eld(:),timest(:)
  REAL(iwp),ALLOCATABLE :: g_coord_pp(:,:,:),ptl_pp(:)
  REAL(iwp),ALLOCATABLE :: prop(:,:),val(:,:)
  INTEGER,ALLOCATABLE   :: rest(:,:),g_num_pp(:,:),g_g_pp(:,:)
  INTEGER,ALLOCATABLE   :: no(:),no_f_pp(:),no_pp(:),no_pp_temp(:)
  INTEGER,ALLOCATABLE   :: sense(:),node(:)
  INTEGER,ALLOCATABLE   :: etype_pp(:)

!------------------------------------------------------------------------------
! 3. Read command line: model_name and instance_id
!------------------------------------------------------------------------------

  ALLOCATE(timest(20)); timest=zero; timest(1)=elap_time()

  CALL find_pe_procs(numpe,npes)

  argc = iargc()
  IF( argc /= 2 ) THEN
     IF(numpe==1) THEN
        PRINT*
        PRINT*, "Usage:  rfemsolve_thermal <model_name> <instance-id>"
        PRINT*
        PRINT*, "        expects:"
        PRINT*, "          <model_name>-<instance-id>.d"
        PRINT*, "          <model_name>-<instance-id>.dat"
        PRINT*, "          <model_name>-<instance-id>.mat"
        PRINT*, "          <model_name>.bnd"
        PRINT*, "          <model_name>.fix"
        PRINT*
     END IF
     CALL MPI_BARRIER(MPI_COMM_WORLD,ier)
     STOP
  END IF

  CALL GETARG(1, job_in)
  CALL GETARG(2, instance_id)

  inst_in = job_in(1:LEN_TRIM(job_in)) // "-" // instance_id(1:LEN_TRIM(instance_id))
  base_in = job_in

!------------------------------------------------------------------------------
! 4. Read control data from <model>-<instance>.dat (rfemsolve format)
!    element mesh partition np_types nels nn nr nip nod loaded fixed tol limit mises
!------------------------------------------------------------------------------

  IF(numpe==1) THEN
     fname = inst_in(1:LEN_TRIM(inst_in)) // ".dat"
     OPEN(10, FILE=fname, STATUS='OLD', ACTION='READ')
     READ(10,*) element,meshgen,partitioner,np_types,nels,nn,nr,nip,nod, &
                loaded_freedoms,fixed_freedoms,tol,limit,mises
     CLOSE(10)
  END IF

  CALL MPI_BCAST(nels,           1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(nn,             1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(nr,             1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(nip,            1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(nod,            1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(loaded_freedoms,1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(fixed_freedoms, 1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(limit,          1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(meshgen,        1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(partitioner,    1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(np_types,       1, MPI_INTEGER,   0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(tol,            1, MPI_REAL8,     0, MPI_COMM_WORLD, ier)
  CALL MPI_BCAST(element,       15, MPI_CHARACTER, 0, MPI_COMM_WORLD, ier)

  nres = 1   ! report on equation 1 (first interior DOF)

  CALL calc_nels_pp(inst_in,nels,npes,numpe,partitioner,nels_pp)

  ndof = nod*nodof
  ntot = ndof

  ALLOCATE(g_num_pp(nod,nels_pp))
  ALLOCATE(g_coord_pp(nod,ndim,nels_pp))
  ALLOCATE(etype_pp(nels_pp))
  ALLOCATE(prop(nprops,np_types))

  g_num_pp   = 0
  g_coord_pp = zero
  etype_pp   = 0
  prop       = zero

  timest(2) = elap_time()

!------------------------------------------------------------------------------
! 5. Read mesh and per-element material type assignments
!------------------------------------------------------------------------------

  CALL read_elements(inst_in,iel_start,nn,npes,numpe,etype_pp,g_num_pp)
  timest(3) = elap_time()

  IF(meshgen == 2) CALL abaqus2sg(element,g_num_pp)
  timest(4) = elap_time()

  CALL read_g_coord_pp(inst_in,g_num_pp,nn,npes,numpe,g_coord_pp)
  timest(5) = elap_time()

!------------------------------------------------------------------------------
! 6. Read per-element diffusivity from <model>-<instance>.mat
!    Format: *MATERIAL nels 1 / kx / iel D_value
!------------------------------------------------------------------------------

  fname = inst_in(1:LEN_TRIM(inst_in)) // ".mat"
  CALL read_materialValue(prop, fname, numpe, npes)

!------------------------------------------------------------------------------
! 7. Read boundary restraints from base model (shared across instances)
!------------------------------------------------------------------------------

  IF(nr>0) THEN
     ALLOCATE(rest(nr,nodof+1)); rest=0
     CALL read_rest(base_in,numpe,rest)
  END IF

!------------------------------------------------------------------------------
! 8. Allocate working arrays
!------------------------------------------------------------------------------

  ALLOCATE(points(nip,ndim),jac(ndim,ndim),storkc_pp(ntot,ntot,nels_pp), &
           deriv(ndim,nod),kcx(ntot,ntot),weights(nip),der(ndim,nod),     &
           pmul_pp(ntot,nels_pp),utemp_pp(ntot,nels_pp),col(ntot,1),eld(ntot), &
           g_g_pp(ntot,nels_pp),kcy(ntot,ntot),row(1,ntot),kcz(ntot,ntot))

!------------------------------------------------------------------------------
! 9. Steering array and equation count
!------------------------------------------------------------------------------

  timest(6) = elap_time()
  g_g_pp = 0
  neq    = 0

  IF(nr>0) THEN
     CALL rearrange_2(rest)
     elements_0: DO iel=1,nels_pp
        CALL find_g4(g_num_pp(:,iel),g_g_pp(:,iel),rest)
     END DO elements_0
  ELSE
     g_g_pp = g_num_pp
  END IF

  neq = MAXVAL(g_g_pp)
  neq = max_p(neq)
  CALL calc_neq_pp
  CALL calc_npes_pp(npes,npes_pp)
  CALL make_ggl(npes_pp,npes,g_g_pp)

  DO i=1,neq_pp
     IF(nres==ieq_start+i-1) THEN; it=numpe; is=i; END IF
  END DO

  IF(numpe==it) THEN
     fname = inst_in(1:LEN_TRIM(inst_in)) // ".res"
     OPEN(11,FILE=fname,STATUS='REPLACE',ACTION='WRITE')
     WRITE(11,'(A,I5,A)') "This job ran on ", npes, "  processes"
     WRITE(11,'(A,3(I12,A))') "There are ",nn," nodes",nr, &
          " restrained and   ",neq," equations"
     WRITE(11,'(A,F10.4)') "Time after setup is ",elap_time()-timest(1)
  END IF

  ALLOCATE(p_pp(neq_pp),r_pp(neq_pp),x_pp(neq_pp),xnew_pp(neq_pp), &
           u_pp(neq_pp),diag_precon_pp(neq_pp),d_pp(neq_pp))
  r_pp=zero; p_pp=zero; x_pp=zero; xnew_pp=zero; diag_precon_pp=zero

  timest(7) = elap_time()

!------------------------------------------------------------------------------
! 10. Per-element stiffness integration using per-element kx/ky/kz = D
!------------------------------------------------------------------------------

  CALL sample(element,points,weights)
  storkc_pp = zero

  elements_1: DO iel=1,nels_pp

     kx = prop(1, etype_pp(iel))
     ky = kx
     kz = kx

     kcx=zero; kcy=zero; kcz=zero

     gauss_pts_1: DO i=1,nip
        CALL shape_der(der,points,i)
        jac = MATMUL(der,g_coord_pp(:,:,iel))
        det = determinant(jac)
        CALL invert(jac)
        deriv = MATMUL(jac,der)
        row(1,:) = deriv(1,:); eld = deriv(1,:); col(:,1) = eld
        kcx = kcx + MATMUL(col,row)*det*weights(i)
        row(1,:) = deriv(2,:); eld = deriv(2,:); col(:,1) = eld
        kcy = kcy + MATMUL(col,row)*det*weights(i)
        row(1,:) = deriv(3,:); eld = deriv(3,:); col(:,1) = eld
        kcz = kcz + MATMUL(col,row)*det*weights(i)
     END DO gauss_pts_1

     storkc_pp(:,:,iel) = kcx*kx + kcy*ky + kcz*kz

  END DO elements_1

  timest(8) = elap_time()

!------------------------------------------------------------------------------
! 11. Diagonal preconditioner
!------------------------------------------------------------------------------

  ALLOCATE(diag_precon_tmp(ntot,nels_pp)); diag_precon_tmp=zero

  elements_1a: DO iel=1,nels_pp
     DO i=1,ndof
        diag_precon_tmp(i,iel) = diag_precon_tmp(i,iel) + storkc_pp(i,i,iel)
     END DO
  END DO elements_1a

  CALL scatter(diag_precon_pp,diag_precon_tmp)
  DEALLOCATE(diag_precon_tmp)

!------------------------------------------------------------------------------
! 12. Read fixed freedoms from base model .fix file
!------------------------------------------------------------------------------

  IF(fixed_freedoms>0) THEN
     ALLOCATE(node(fixed_freedoms),no_pp_temp(fixed_freedoms), &
              val_f(fixed_freedoms),no(fixed_freedoms),sense(fixed_freedoms))
     node=0; no=0; no_pp_temp=0; sense=0; val_f=zero

     CALL read_fixed(base_in,numpe,node,sense,val_f)
     CALL find_no2(g_g_pp,g_num_pp,node,sense,no)
     CALL reindex(ieq_start,no,no_pp_temp,fixed_freedoms_pp, &
                  fixed_freedoms_start,neq_pp)
     ALLOCATE(no_f_pp(fixed_freedoms_pp),store_pp(fixed_freedoms_pp))
     no_f_pp=0; store_pp=zero
     no_f_pp = no_pp_temp(1:fixed_freedoms_pp)
     DEALLOCATE(node,no,sense,no_pp_temp)
  END IF

  IF(fixed_freedoms==0) fixed_freedoms_pp=0
  IF(nr>0) DEALLOCATE(rest)

!------------------------------------------------------------------------------
! 13. Read loaded freedoms (optional — typically zero for diffusion)
!------------------------------------------------------------------------------

  IF(loaded_freedoms>0) THEN
     ALLOCATE(node(loaded_freedoms),val(nodof,loaded_freedoms), &
              no_pp_temp(loaded_freedoms))
     val=zero; node=0
     CALL read_loads(inst_in,numpe,node,val)
     CALL reindex(ieq_start,node,no_pp_temp,loaded_freedoms_pp, &
                  loaded_freedoms_start,neq_pp)
     ALLOCATE(no_pp(loaded_freedoms_pp))
     no_pp = no_pp_temp(1:loaded_freedoms_pp)
     DO i=1,loaded_freedoms_pp
        r_pp(no_pp(i)-ieq_start+1) = val(1,loaded_freedoms_start+i-1)
     END DO
     q = SUM_P(r_pp)
     DEALLOCATE(node,val,no_pp_temp,no_pp)
  END IF

!------------------------------------------------------------------------------
! 14. Invert preconditioner with penalty for fixed freedoms
!------------------------------------------------------------------------------

  IF(fixed_freedoms_pp>0) THEN
     DO i=1,fixed_freedoms_pp
        j                 = no_f_pp(i) - ieq_start + 1
        diag_precon_pp(j) = diag_precon_pp(j) + penalty
        store_pp(i)       = diag_precon_pp(j)
     END DO
  END IF

  diag_precon_pp = 1._iwp/diag_precon_pp

!------------------------------------------------------------------------------
! 15. Initialise PCG with penalty RHS for fixed freedoms
!------------------------------------------------------------------------------

  IF(fixed_freedoms_pp>0) THEN
     DO i=1,fixed_freedoms_pp
        j       = no_f_pp(i) - ieq_start + 1
        k       = fixed_freedoms_start + i - 1
        r_pp(j) = store_pp(i) * val_f(k)
     END DO
  END IF

  d_pp = diag_precon_pp*r_pp
  p_pp = d_pp
  x_pp = zero

  timest(9) = elap_time()

!------------------------------------------------------------------------------
! 16. PCG iterations
!------------------------------------------------------------------------------

  iters = 0

  iterations: DO
     iters    = iters + 1
     u_pp     = zero
     pmul_pp  = zero
     utemp_pp = zero

     CALL gather(p_pp,pmul_pp)
     elements_2: DO iel=1,nels_pp
        utemp_pp(:,iel) = MATMUL(storkc_pp(:,:,iel),pmul_pp(:,iel))
     END DO elements_2
     CALL scatter(u_pp,utemp_pp)

     IF(fixed_freedoms_pp>0) THEN
        DO i=1,fixed_freedoms_pp
           j       = no_f_pp(i) - ieq_start + 1
           u_pp(j) = p_pp(j) * store_pp(i)
        END DO
     END IF

     up      = DOT_PRODUCT_P(r_pp,d_pp)
     alpha   = up / DOT_PRODUCT_P(p_pp,u_pp)
     xnew_pp = x_pp + p_pp*alpha
     r_pp    = r_pp - u_pp*alpha
     d_pp    = diag_precon_pp*r_pp
     beta    = DOT_PRODUCT_P(r_pp,d_pp)/up
     p_pp    = d_pp + p_pp*beta

     CALL checon_par(xnew_pp,tol,converged,x_pp)
     IF(converged .OR. iters==limit) EXIT

  END DO iterations

  timest(10) = elap_time()

  DEALLOCATE(p_pp,r_pp,x_pp,u_pp,d_pp,diag_precon_pp,storkc_pp,pmul_pp)

!------------------------------------------------------------------------------
! 17. Write summary to .res
!------------------------------------------------------------------------------

  IF(numpe==it) THEN
     WRITE(11,'(A,I5)')   "The number of iterations to convergence was ",iters
     WRITE(11,'(A,F10.4)')"Integration and preconditioning took ", &
          timest(9)-timest(8)
     WRITE(11,'(A,F10.4)')"Time spent in the solver was ",timest(10)-timest(9)
  END IF

!------------------------------------------------------------------------------
! 18. Output scalar field (concentration / temperature) for ParaView
!------------------------------------------------------------------------------

  CALL calc_nodes_pp(nn,npes,numpe,node_end,node_start,nodes_pp)

  IF(numpe==1) THEN
     WRITE(ch,'(I6.6)') numpe
     fname = inst_in(1:LEN_TRIM(inst_in)) // ".ensi.NDPTL-" // ch
     OPEN(12,FILE=fname,STATUS='REPLACE',ACTION='WRITE')
     WRITE(12,'(A)') "Alya Ensight Gold --- Scalar per-node variable file"
     WRITE(12,'(A/A/A)') "part","    1","coordinates"
  END IF

  ALLOCATE(ptl_pp(nodes_pp*ndim))
  ptl_pp   = zero
  utemp_pp = zero
  CALL gather(xnew_pp(1:),utemp_pp)
  CALL scatter_nodes(npes,nn,nels_pp,g_num_pp,nod,nodof,nodes_pp, &
       node_start,node_end,utemp_pp,ptl_pp,1)
  CALL dismsh_ensi_p(12,1,nodes_pp,npes,numpe,1,ptl_pp)

  timest(11) = elap_time()

  IF(numpe==it) THEN
     WRITE(11,'(A,F10.4)') "This analysis took ",elap_time()-timest(1)
     CLOSE(11)
  END IF

  IF(numpe==1) CLOSE(12)

  CALL SHUTDOWN()

END PROGRAM rfemsolve_thermal
