"""Figure R1: Dragt surface of section (z = 0), PS (20 steps/T) vs Boris at 65/130/260/650 steps/T, particle_322bba.yml.
PS is drawn first, Boris on top: any visible PS (blue) is part of the true orbit that Boris never visits.
Panel titles give the fraction of the PS section curve with no Boris crossing within 1% of the arc size."""
import os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import boris_section as bs
H=os.path.dirname(os.path.abspath(__file__)); O=os.path.join(H,'out')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':12,'mathtext.fontset':'cm'})
C_PS='#1F5FBF'; C_B='#FF1493'
P=bs.setup_from_yml(os.path.join(H,'particle_322bba.yml'),65,None); s0=P['state0']
L,W,Pp=bs.dragt_frame(s0,1.0); rb,rdb=bs.boundary(W,Pp,1.0)
rho0=np.hypot(*s0[:2]); launch=(rho0/L,((s0[0]*s0[3]+s0[1]*s0[4])/rho0)*L**2)
def sec(nm):
    c=np.load(os.path.join(O,nm+'.npz'))['crossings']; return bs.to_section(c,L)
pr,pd=bs.to_section(np.load(os.path.join(O,'ps20_hermite.npy')),L)   # PS at 20 steps/T, Hermite crossings (hermite_check.py)
tp=np.c_[(pr-1)/0.03,pd/0.03]; tp=np.c_[(pr-1)/0.03,pd/0.03]
fig,axs=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True); axs=axs.ravel()
for ax,n,l in zip(axs,(65,130,260,650),'abcd'):
    br,bd=sec(f'boris{n}'); dist,_=cKDTree(np.c_[(br-1)/0.03,bd/0.03]).query(tp); miss=100*np.mean(dist>0.01)
    ax.plot(rb,rdb,'k-',lw=1.2); ax.plot(rb,-rdb,'k-',lw=1.2)
    ax.plot(pr,pd,'.',ms=4,mew=0,color=C_PS,label='PS (20 steps per $T$)',rasterized=True)
    ax.plot(br,bd,'.',ms=2,mew=0,color=C_B,label=f'Boris ({n} steps per $T$)',rasterized=True)
    ax.plot(*launch,'D',mfc='white',mec='black',ms=6,label='launch')
    ax.set_title(f'({l}) Boris, {n} steps per $T$: {miss:.0f}% of PS curve not visited',fontsize=11)
    ax.grid(alpha=0.3,lw=0.5)
    from matplotlib.lines import Line2D
    h=[Line2D([],[],ls='',marker='o',ms=6,color=C_PS,label='PS (20 steps per $T$)'),Line2D([],[],ls='',marker='o',ms=6,color=C_B,label=f'Boris ({n} steps per $T$)'),Line2D([],[],ls='',marker='D',mfc='white',mec='black',ms=6,label='launch')]
    ax.legend(handles=h,loc='center',fontsize=9,frameon=False)
    ax.set_xlim(0.972,1.034); ax.set_ylim(-0.034,0.034)
for ax in axs[2:]: ax.set_xlabel(r'$\rho$')
for ax in axs[::2]: ax.set_ylabel(r'$\dot\rho$')
fig.tight_layout(); out=os.path.join(H,'FigR1_section_overlay.png'); fig.savefig(out,dpi=300); print(out)
