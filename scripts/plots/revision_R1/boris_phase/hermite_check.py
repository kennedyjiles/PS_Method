"""Check that the PS20 vs PS260 mu offset at crossings is crossing-finder (linear interpolation) error:
re-find PS20 crossings with cubic Hermite interpolation (positions with v, velocities with a = v x B)."""
import os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../../../..')))
from ps_method import dipoleb_physics as dp
import boris_section as bs
def acc(X):
    x,y,z,vx,vy,vz=X; r2=x*x+y*y+z*z; r5=r2**-2.5; Bx=-3*x*z*r5; By=-3*y*z*r5; Bz=-(3*z*z-r2)*r5
    return np.array([vy*Bz-vz*By, vz*Bx-vx*Bz, vx*By-vy*Bx])
def herm(p0,p1,m0,m1,h,s):
    h00=2*s**3-3*s**2+1; h10=s**3-2*s**2+s; h01=-2*s**3+3*s**2; h11=s**3-s**2
    return h00*p0+h10*h*m0+h01*p1+h11*h*m1
n=float(sys.argv[1]); P=bs.setup_from_yml(os.path.join(HERE,'particle_322bba.yml'),n,None)
d=P['state0'].copy(); dt=P['dt']; N=P['steps']; spg=int(round(P['T_gyro']/dt)); rows=[]; k=0
while k<N:
    m=min(spg*300,N-k); h,o=dp.ps_integrate(40,m,d,np.finfo(float).eps,1.0,dt); a=h[0:6]
    idx=np.where(a[2,:-1]*a[2,1:]<0)[0]
    for i in idx:
        X0=a[:,i]; X1=a[:,i+1]; A0=acc(X0); A1=acc(X1)
        lo,hi=0.0,1.0
        for _ in range(60):
            s=0.5*(lo+hi); zs=herm(X0[2],X1[2],X0[5],X1[5],dt,s)
            if zs*X0[2]>0: lo=s
            else: hi=s
        s=0.5*(lo+hi); pos=herm(X0[:3],X1[:3],X0[3:],X1[3:],dt,s); vel=herm(X0[3:],X1[3:],A0,A1,dt,s)
        rows.append([(k+i+s)*dt,pos[0],pos[1],vel[0],vel[1],vel[2]])
    d=h[0:6,-1].copy(); k+=m
np.save(os.path.join(HERE,'out',f'ps{n:g}_hermite.npy'),np.array(rows)); print(len(rows))
