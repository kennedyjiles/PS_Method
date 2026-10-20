import json, os, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
sys.path.insert(0, os.environ["PS_ROOT"]); from ps_method.utils import plt_config
plt_config(scale=1.4)
if not any("Times New Roman" in f.name for f in fm.fontManager.ttflist): plt.rcParams["font.family"]="Liberation Serif"
d=json.load(open("ke3.json")); C={"rk45":"#E69F00","ps":"#009E73"}; L={"rk45":"RK45","ps":"PS13"}
for tag,rid in [("a_rtol1e-11","619028"),("b_rtol1e-15","6f101d"),("c_rtol1e-18","d6d2ff")]:
    fig,ax=plt.subplots(figsize=(6,5))
    for g in ["rk45","ps"]:
        x=np.array(d[rid][g]["x"]); y=np.array(d[rid][g]["y"]); m=(x>0)&(y>0)
        ax.plot(x[m],y[m],color=C[g],lw=1.2); ax.text(x[m][-1]*1.15,y[m][-1],L[g],color=C[g],va="center",fontsize=16)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-17,1e-2); ax.set_xlim(1e-2,1.5e5*1.0)
    ax.set_xlabel(r"$\tau/T$"); ax.set_ylabel(r"$|\Delta E|/E_0$")
    ax.grid(True,which="major",ls="--",lw=0.7,alpha=0.7)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"R2_ke_{tag}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
