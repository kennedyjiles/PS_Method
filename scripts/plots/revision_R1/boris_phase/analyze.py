import numpy as np, os
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'out')
def load(t): return np.load(os.path.join(O,t+'.npz'))['crossings']
T=2*np.pi*2.6812803139531813**3
def q(c):
    x,y,vx,vy,vz=c[:,1],c[:,2],c[:,3],c[:,4],c[:,5]; rho=np.hypot(x,y)
    vr=(x*vx+y*vy)/rho; vp=(x*vy-y*vx)/rho
    return dict(t=c[:,0]/T, psi=np.arctan2(vp,vr), mu=(vx**2+vy**2)*rho**3/2, az=np.arctan2(y,x), up=np.sign(vz))
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
ref=q(load('ps260'))
names=['ps65','ps130','ps130_pert','adaptive_ps65','adaptive_ps130','boris65','boris130','boris260','boris650']
print('ref crossings',len(ref['t']),'last t %.0f'%ref['t'][-1])
for nm in names:
    c=q(load(nm)); m=min(len(c['t']),len(ref['t']))
    same_dir=np.all(c['up'][:m]==ref['up'][:m])
    dpsi=np.unwrap(c['psi'][:m]-ref['psi'][:m]); dt=(c['t'][:m]-ref['t'][:m]); dmu=np.abs(c['mu'][:m]/ref['mu'][:m]-1); daz=np.unwrap(c['az'][:m]-ref['az'][:m])
    tt=ref['t'][:m]; sl=np.polyfit(tt,dpsi,1)[0]
    pts=[np.searchsorted(tt,g) for g in (1e2,1e3,1e4,np.inf)]
    s=' | '.join(f'T={tt[min(i,m-1)]:.0e}: dpsi {dpsi[min(i,m-1)]:+.2e} dt {dt[min(i,m-1)]:+.1e}T dmu {dmu[min(i,m-1)]:.1e} daz {daz[min(i,m-1)]:+.1e}' for i in pts)
    print(f'{nm:15s} n={len(c["t"])} dirOK={same_dir} slope dpsi/dT={sl:+.3e} rad/T\n   {s}')
