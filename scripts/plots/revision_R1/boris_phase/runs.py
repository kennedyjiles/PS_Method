"""
Phase-drift study for Figure R1 (Boris vs PS), particle_322bba.yml.
Uses the paper's constants and dt rounding via boris_section.setup_from_yml (identical ICs to run 322bba).
  python runs.py ps <n> [perturb]   fixed-step PS (ps_integrate, adaptive order) -> out/ps<n>[_pert].npz
  python runs.py boris <n>          Boris (boris_section.boris_chunk, pipeline arithmetic) -> out/boris<n>.npz
  python runs.py adaptive <h5> <tag> crossings from an existing run.py PS h5 (adaptive stepping) -> out/<tag>.npz
Each .npz holds equatorial crossings (t,x,y,vx,vy,vz) plus diagnostics.
"""
import os, sys, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../../../..')))
import boris_section as bs
OUT=os.path.join(HERE,os.environ.get('POUT','out')); os.makedirs(OUT,exist_ok=True)
YML=os.path.join(HERE,os.environ.get('PYML','particle_322bba.yml'))

def crossings_from(a,k0,dt):
    z=a[2]; idx=np.where(z[:-1]*z[1:]<0)[0]
    if not len(idx): return np.empty((0,6))
    tf=np.abs(z[idx])/(np.abs(z[idx])+np.abs(z[idx+1])); out=np.empty((len(idx),6)); out[:,0]=(k0+idx+tf)*dt
    for c,r in zip(range(1,6),(0,1,3,4,5)): out[:,c]=a[r,idx]+tf*(a[r,idx+1]-a[r,idx])
    return out

def pphi(a): r=np.sqrt((a[0:3]**2).sum(0)); return (a[0]*a[4]-a[1]*a[3])-(a[0]**2+a[1]**2)/r**3

def run_ps(n,perturb=False):
    from ps_method import dipoleb_physics as dp
    P=bs.setup_from_yml(YML,n,None); d=P['state0'].copy()
    if perturb: d[0]*=1+1e-12
    dt=P['dt']; N=P['steps']; spg=int(round(P['T_gyro']/dt)); chunk=spg*300
    rows=[]; k=0; osum=0; omax=0; ncap=0; p0=pphi(d[:,None])[0]; E0=0.5*d[3:]@d[3:]; pe=0; ee=0; rmin=9
    while k<N:
        m=min(chunk,N-k); h,o=dp.ps_integrate(40,m,d,np.finfo(float).eps,1.0,dt)
        osum+=o[1:].sum(); omax=max(omax,int(o.max())); ncap+=int((o[1:]>=40).sum())
        a=h[0:6]; rows.append(crossings_from(a,k,dt))
        pe=max(pe,np.abs(pphi(a)/p0-1).max()); ee=max(ee,np.abs(0.5*(a[3:]**2).sum(0)/E0-1).max()); rmin=min(rmin,np.sqrt((a[0:3]**2).sum(0)).min())
        d=h[0:6,-1].copy(); k+=m
    tag=f'ps{n:g}'+('_pert' if perturb else '')
    np.savez(os.path.join(OUT,tag+'.npz'),crossings=np.concatenate(rows),dt=dt,T_gyro=P['T_gyro'],mean_order=osum/N,max_order=omax,
             n_order_cap=ncap,max_dPphi=pe,max_dE=ee,r_min=rmin,steps=N)
    print(tag,dict(dt=dt,mean_order=round(osum/N,2),max_order=omax,n_cap=ncap,max_dPphi=f'{pe:.1e}',max_dE=f'{ee:.1e}',r_min=round(rmin,5)))

def run_boris(n):
    P=bs.setup_from_yml(YML,n,None); state=P['state0'].copy(); dt=P['dt']; N=P['steps']
    cbuf=np.empty((2_000_000,6)); parts=[]; done=0; pmn=1e300; pmx=-1e300; r2m=1e300
    while done<N:
        m=min(2_000_000,N-done); nc,dn,r2,pa,pb,h=bs.boris_chunk(state,dt,m,1.0,P['r_atm']**2,False,done,cbuf)
        parts.append(cbuf[:nc].copy()); done+=dn; pmn=min(pmn,pa); pmx=max(pmx,pb); r2m=min(r2m,r2)
    s0=P['state0']; p0=pphi(s0[:,None])[0]
    np.savez(os.path.join(OUT,f'boris{n:g}.npz'),crossings=np.concatenate(parts),dt=dt,T_gyro=P['T_gyro'],Pphi_rel_range=(pmx-pmn)/abs(p0),r_min=np.sqrt(r2m),steps=N)
    print(f'boris{n:g}',dict(dt=dt,Pphi_rel_range=f'{(pmx-pmn)/abs(p0):.1e}',r_min=round(float(np.sqrt(r2m)),5)))

def run_adaptive(h5,tag):
    c,dt=bs.ps_crossings(h5); np.savez(os.path.join(OUT,tag+'.npz'),crossings=c,dt=dt); print(tag,len(c),dt)

if __name__=='__main__':
    w=sys.argv[1]
    if w=='ps': run_ps(float(sys.argv[2]),len(sys.argv)>3)
    elif w=='boris': run_boris(float(sys.argv[2]))
    elif w=='adaptive': run_adaptive(sys.argv[2],sys.argv[3])
