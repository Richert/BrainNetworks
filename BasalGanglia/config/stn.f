
	subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)
	implicit none
	integer, intent(in) :: ndim, icp(*), ijac
	double precision, intent(in) :: y(ndim), args(*)
	double precision, intent(out) :: y_delta(ndim)
	double precision, intent(inout) :: dfdu(ndim,ndim), 
     & dfdp(ndim,*)
	double precision R_e,V_e,delta_e,tau_e,eta_e,J_ee,u

	! declare constants
	delta_e = args(1)
	tau_e = args(2)
	eta_e = args(3)
	J_ee = args(4)
	u = args(5)

	! extract state variables from input vector
	R_e = y(1)
	V_e = y(2)

	! calculate right-hand side update of equation system
	y_delta(1) = delta_e / (3.141592653589793 * tau_e ** 2) 
     & + (2.0 * R_e * V_e) / tau_e
	y_delta(2) = (V_e ** 2 + eta_e + u) / tau_e + J_ee * R_e 
     & - tau_e * (3.141592653589793 * R_e) ** 2

	end subroutine func
 
 
	subroutine stpnt(ndim, y, args, t)
	implicit None
	integer, intent(in) :: ndim
	double precision, intent(inout) :: y(ndim), args(*)
	double precision, intent(in) :: T
	double precision R_e,V_e,delta_e,tau_e,eta_e,J_ee,u


	delta_e = 1.0499999523162842
	tau_e = 6.0
	eta_e = -3.0
	J_ee = 1.0
	u = 0.0


	args(1) = delta_e
	args(2) = tau_e
	args(3) = eta_e
	args(4) = J_ee
	args(5) = u


	y(1) = 0.016092609614133835
	y(2) = -1.7307393550872803


	end subroutine stpnt

	subroutine bcnd
	end subroutine bcnd

	subroutine icnd
	end subroutine icnd

	subroutine fopt
	end subroutine fopt

	subroutine pvls
	end subroutine pvls

