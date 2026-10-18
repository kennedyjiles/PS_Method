"""Direct same-time gyrophase comparison: Boris vs PS at the SAME dt (paper rounding), first G gyroperiods.
Gyrophase = angle of v_perp in a local frame (e1 = unit(B x z-hat x B)... see below), unwrapped per step."""
import os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../../../..')))
from ps_method import dipoleb_physics as dp
import boris_section as bs
YML=os.path.join(HERE,'particle_322bba.yml')
def Bf(x,y,z):
    r2=x*x+y*y+z*z; r5=r2**-2.5; return np.array([-3*x*z*r5,-3*y*z*r5,-(3*z*z-r2)*r5])
def gphase(X):
    x,y,z,vx,vy,vz=X; B=Bf(x,y,z); b=B/np.linalg.norm(B,axis=0)
    phi_hat=np.array([-y,x,np.zeros_like(x)])/np.hypot(x,y)          # azimuthal unit vector (perp to dipole B everywhere)
    e2=np.cross(b.T,phi_hat.T).T                                      # completes right-handed (phi_hat, e2, b)
    v=np.array([vx,vy,vz]); vpar=(v*b).sum(0); vperp=v-vpar*b
    return np.unwrap(np.arctan2((vperp*e2).sum(0),(vperp*phi_hat).sum(0)))
G=float(sys.argv[2]) if len(sys.argv)>2 else 2000
n=float(sys.argv[1]); P=bs.setup_from_yml(YML,n,None); dt=P['dt']; T=P['T_gyro']; N=int(G*T/dt)
h,o=dp.ps_integrate(40,N,P['state0'].copy(),np.finfo(float).eps,1.0,dt); ps=h[0:6]
bo=dp.boris_fixed_step(P['state0'].copy(),dt,N,1.0)
t=np.arange(N+1)*dt/T; g1=gphase(ps); g2=gphase(bo)
d=g2-g1; idx=[np.searchsorted(t,g) for g in (1,10,100,1000,G-1)]
sl=np.polyfit(t,d,1)[0]
print(f'n={n:g} dt={dt}: gyrophase(Boris)-gyrophase(PS) at T=',' '.join(f'{t[i]:.0f}:{d[i]:+.3e}' for i in idx),f'| fit slope {sl:+.3e} rad/T | PS gyros/T check {(g1[-1]-g1[0])/(2*np.pi*t[-1]):.4f}')
np.savez(os.path.join(HERE,'out',f'gyrophase_n{n:g}.npz'),t=t[::max(1,int(n)//13)],d=d[::max(1,int(n)//13)],dt=dt)
