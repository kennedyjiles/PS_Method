"""Figure R1 (three panels) from out/*.npz produced by runs.py and gyrophase.py."""
import os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__)); O=os.path.join(H,'out')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':11,'mathtext.fontset':'cm'})
T=2*np.pi*2.6812803139531813**3
CPS='#009E73'; CB={65:'#D55E00',130:'#CC79A7',260:'#0072B2',650:'#555555'}
def cr(nm):
    c=np.load(os.path.join(O,nm+'.npz'))['crossings']; rho=np.hypot(c[:,1],c[:,2]); return c[:,0]/T,(c[:,3]**2+c[:,4]**2)*rho**3/2
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
# (a) gyrophase error at identical times
for n in (65,130,260,650):
    g=np.load(os.path.join(O,f'gyrophase_n{n}.npz')); m=g['t']>0
    ax[0].loglog(g['t'][m],np.abs(g['d'][m]),color=CB[n],lw=1,label=f'Boris {n}')
ax[0].set_xlabel(r'$\tau/T$'); ax[0].set_ylabel(r'$|\psi_{\rm Boris}-\psi_{\rm PS}|$ (rad)'); ax[0].set_xlim(1,2e3); ax[0].set_ylim(1e-5,2e1)
ax[0].legend(fontsize=8,loc='lower right',title='steps per $T$',title_fontsize=8)
# (b) bounce-phase (equatorial crossing time) error
t0,_=cr('ps260')
for nm,col,lab,ls in [('ps65',CPS,'PS 65','-'),('ps130',CPS,'PS 130','--')]+[(f'boris{n}',CB[n],f'Boris {n}','-') for n in (65,130,260,650)]:
    t,_=cr(nm); k=min(len(t),len(t0)); d=np.abs(t[:k]-t0[:k]); d[d==0]=np.nan
    ax[1].loglog(t0[:k],d,color=col,ls=ls,lw=1.2 if nm.startswith('ps') else 1,label=lab)
ax[1].set_xlabel(r'$\tau/T$'); ax[1].set_ylabel(r'$|\Delta\tau_{\rm cross}|/T$ vs. PS 260'); ax[1].set_xlim(1,1e5); ax[1].legend(fontsize=8,ncol=2,loc='lower right')
# (c) distribution of mu at equatorial crossings
_,m0=cr('ps260'); M0=m0[0]; bins=np.linspace(0.38,1.18,81)
ax[2].hist(m0/M0,bins,density=True,histtype='stepfilled',color=CPS,alpha=0.35,label='PS 65/130/260')
for n in (65,130,260,650):
    _,m=cr(f'boris{n}'); ax[2].hist(m/M0,bins,density=True,histtype='step',color=CB[n],lw=1.3,label=f'Boris {n}')
ax[2].set_xlabel(r'$\mu/\mu_0$ at equatorial crossings'); ax[2].set_ylabel('probability density'); ax[2].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0.06,0.99))
for a,l in zip(ax,'abc'): a.text(0.02,0.97,f'({l})',transform=a.transAxes,va='top',fontsize=12); a.grid(alpha=0.3,which='both',lw=0.4)
fig.tight_layout(); out=os.path.join(H,'FigR1_boris_phase.png'); fig.savefig(out,dpi=300); print(out)
