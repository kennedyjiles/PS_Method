"""Invariant errors vs time (max per gyroperiod) for PS20 and Boris 65/130/260/650 -> out/inv_*.npz; then plot four panels.
  python invariants.py compute | plot"""
import os, sys, numpy as np, numba
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../../../..')))
import boris_section as bs
YML=os.path.join(HERE,'particle_322bba.yml'); O=os.path.join(HERE,'out')

@numba.njit(cache=True)
def boris_inv(d,dt,N,spg_steps_per_bin,nbin):
    x,y,z,vx,vy,vz=d; E0=0.5*(vx*vx+vy*vy+vz*vz); r=np.sqrt(x*x+y*y+z*z); P0=(x*vy-y*vx)-(x*x+y*y)/r**3
    eE=np.zeros(nbin); eP=np.zeros(nbin)
    for i in range(N):
        xh=x+0.5*dt*vx; yh=y+0.5*dt*vy; zh=z+0.5*dt*vz
        r2=xh*xh+yh*yh+zh*zh; r5=r2**-2.5
        Bx=-3.0*xh*zh*r5; By=-3.0*yh*zh*r5; Bz=-(3.0*zh*zh-r2)*r5
        tx=0.5*dt*Bx; ty=0.5*dt*By; tz=0.5*dt*Bz
        f=2.0/(1.0+tx*tx+ty*ty+tz*tz); sx=f*tx; sy=f*ty; sz=f*tz
        vpx=vx+(vy*tz-vz*ty); vpy=vy+(vz*tx-vx*tz); vpz=vz+(vx*ty-vy*tx)
        vx=vx+(vpy*sz-vpz*sy); vy=vy+(vpz*sx-vpx*sz); vz=vz+(vpx*sy-vpy*sx)
        x=xh+0.5*dt*vx; y=yh+0.5*dt*vy; z=zh+0.5*dt*vz
        b=(i+1)//spg_steps_per_bin
        if b<nbin:
            r=np.sqrt(x*x+y*y+z*z)
            e=abs(0.5*(vx*vx+vy*vy+vz*vz)/E0-1.0); p=abs(((x*vy-y*vx)-(x*x+y*y)/r**3)/P0-1.0)
            if e>eE[b]: eE[b]=e
            if p>eP[b]: eP[b]=p
    return eE,eP

def compute():
    from ps_method import dipoleb_physics as dp
    for n in (65,130,260,650):
        P=bs.setup_from_yml(YML,n,None); dt=P['dt']; N=P['steps']; per=int(round(P['T_gyro']/dt)); nb=N//per
        eE,eP=boris_inv(P['state0'].copy(),dt,N,per,nb); np.savez(os.path.join(O,f'inv_boris{n}.npz'),eE=eE,eP=eP,t=np.arange(1,nb+1)*per*dt/P['T_gyro'])
        print('boris',n,'max dE %.1e max dPphi %.1e'%(eE.max(),eP.max()))
    P=bs.setup_from_yml(YML,20,None); dt=P['dt']; N=P['steps']; per=int(round(P['T_gyro']/dt)); nb=N//per
    d=P['state0'].copy(); E0=0.5*d[3:]@d[3:]; r=np.linalg.norm(d[:3]); P0=(d[0]*d[4]-d[1]*d[3])-(d[0]**2+d[1]**2)/r**3
    eE=np.zeros(nb); eP=np.zeros(nb); k=0
    while k<N:
        m=min(per*300,N-k); h,o=dp.ps_integrate(40,m,d,np.finfo(float).eps,1.0,dt); a=h[0:6,1:]
        rr=np.sqrt((a[0:3]**2).sum(0)); e=np.abs(0.5*(a[3:]**2).sum(0)/E0-1); p=np.abs(((a[0]*a[4]-a[1]*a[3])-(a[0]**2+a[1]**2)/rr**3)/P0-1)
        b=(np.arange(k+1,k+m+1))//per; ok=b<nb; np.maximum.at(eE,b[ok],e[ok]); np.maximum.at(eP,b[ok],p[ok])
        d=h[0:6,-1].copy(); k+=m
    np.savez(os.path.join(O,'inv_ps20.npz'),eE=eE,eP=eP,t=np.arange(1,nb+1)*per*dt/P['T_gyro']); print('ps20 max dE %.1e max dPphi %.1e'%(eE.max(),eP.max()))

def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':12,'mathtext.fontset':'cm'})
    C_PS='#1F5FBF'; C_B='#FF1493'; ps=np.load(os.path.join(O,'inv_ps20.npz'))
    miss={65:52,130:28,260:45,650:0}
    fig,axs=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True); axs=axs.ravel()
    for ax,n,l in zip(axs,(65,130,260,650),'abcd'):
        b=np.load(os.path.join(O,f'inv_boris{n}.npz'))
        def pos(v): v=v.copy(); v[v<=0]=np.nan; return v
        ax.loglog(ps['t'],pos(ps['eP']),color=C_PS,lw=1.0,ls='-',label=r'PS (20 steps per $T$): $|\Delta P_\phi|/|P_{\phi0}|$')
        ax.loglog(ps['t'],pos(ps['eE']),color=C_PS,lw=1.0,ls=':',label=r'PS: $|\Delta E|/E_0$')
        ax.loglog(b['t'],pos(b['eP']),color=C_B,lw=1.0,ls='-',label=rf'Boris ({n} steps per $T$): $|\Delta P_\phi|/|P_{{\phi0}}|$')
        ax.loglog(b['t'],pos(b['eE']),color=C_B,lw=1.0,ls=':',label=r'Boris: $|\Delta E|/E_0$')
        ax.set_title(f'({l}) Boris, {n} steps per $T$ ({miss[n]}% of PS section curve not visited)',fontsize=11)
        ax.grid(alpha=0.3,which='both',lw=0.4); ax.set_ylim(1e-17,1e-3); ax.set_xlim(1,1e5)
        ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(0.0,0.42),frameon=False)
    for ax in axs[2:]: ax.set_xlabel(r'$\tau/T$')
    for ax in axs[::2]: ax.set_ylabel('max relative error per gyroperiod')
    fig.tight_layout(); out=os.path.join(HERE,'FigR1_invariants.png'); fig.savefig(out,dpi=300); print(out)

if __name__=='__main__': {'compute':compute,'plot':plot}[sys.argv[1]]()
