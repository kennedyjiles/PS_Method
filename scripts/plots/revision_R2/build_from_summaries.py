# Build scatterplot CSVs from each run's <id>_summary.txt "tail average" blocks
# (final 1% of the run) -- the same source as Tables B.1/B.2.
import glob, os, re, sys, pandas as pd
DATA = sys.argv[1]
def lab(ev):
    ev=float(ev); v,u=(ev/1e3,"keV") if ev<1e6 else (ev/1e6,"MeV")
    return f"{int(round(v))} {u}" if abs(v-round(v))<1e-9 else f"{v:g} {u}"
for group,out in [("protons","proton_summary_results.csv"),("electrons","electron_summary_results.csv")]:
    rows=[]
    for p in sorted(glob.glob(os.path.join(DATA,group,"*","*","*_summary.txt"))):
        t=open(p).read()
        E=float(re.search(r"energy_eV = ([0-9.eE+]+)",t).group(1)); pitch=float(re.search(r"pitch_deg = ([0-9.]+)",t).group(1))
        tm=dict(re.findall(r"^\s{4}(ps|rk4|rk45|rkg) = ([0-9.eE+-]+)",t.split("timing:")[1].split("ps:")[0],re.M))
        def block(name):
            m=re.search(re.escape(name)+r".*?\n(.*?)\n\n",t,re.S); d={}
            for k,mean,mx in re.findall(r"^\s+(\w+)\s*: mean = ([0-9.eE+-]+), max = ([0-9.eE+-]+)",m.group(1),re.M): d[k]=(float(mean),float(mx))
            return d
        dE=block("|delta E|/E0 (tail average)"); dP=block("|delta P_phi|/|P_phi_0| (tail average)")
        for meth in ["RK45","RK4","RKG","PS"]:
            if meth in dE and meth.lower() in tm:
                rows.append(dict(energy_label=lab(E),energy_eV=E,pitch_deg=int(pitch),method=meth,
                    errE_mean=dE[meth][0],errE_max=dE[meth][1],errPphi_mean=dP[meth][0],errPphi_max=dP[meth][1],
                    runtime_s=float(tm[meth.lower()]),src=os.path.basename(p)))
    df=pd.DataFrame(rows).sort_values(["energy_eV","pitch_deg","method"],ascending=[True,False,True])
    df.to_csv(out,index=False); print(group,len(df)); print(df.drop(columns=["src","energy_eV"]).to_string(index=False))
