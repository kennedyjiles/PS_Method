import h5py, numpy as np, json, sys
D=sys.argv[1]; out={}
T=2*np.pi*5.0**3; dt=12.1
for rid in ["619028","6f101d","d6d2ff"]:
    f=h5py.File(f"{D}/{rid}_full.h5","r"); out[rid]={}
    for g in ["rk45","ps"]:
        v=f[f"{g}/y"][3:6,:]; ke=0.5*(v*v).sum(0); e=np.abs(ke-ke[0])/ke[0]
        n=len(e); tau=np.arange(n)*dt/T
        edges=np.unique(np.logspace(0,np.log10(n-1),2500).astype(int))
        xs=[];ys=[]
        for a,b in zip(edges[:-1],edges[1:]):
            seg=e[a:b]; i=a+int(np.argmax(seg)); xs.append(tau[i]); ys.append(float(seg.max()))
        out[rid][g]=dict(x=xs,y=ys)
    print(rid, out[rid]["rk45"]["y"][-1], out[rid]["ps"]["y"][-1]); 
    # value near tau/T=100
    for g in ["rk45","ps"]:
        x=np.array(out[rid][g]["x"]); y=np.array(out[rid][g]["y"]); k=np.argmin(abs(x-100)); print("  ",g,"@100:",y[k])
json.dump(out,open("ke3.json","w"))
