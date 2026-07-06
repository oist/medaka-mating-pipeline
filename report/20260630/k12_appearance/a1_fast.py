import numpy as np, pandas as pd, sys
NEW=r"C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv"
OLD=r"Z:/StephensU/medaka/reconstruction_3d/round5_260611/20251225_K_1_2_FCMC/round5_260611_3d_identified.csv"
OUT=r"report/20260630/k12_appearance/backbone_divergence.csv"
def log(*a): print(*a); sys.stdout.flush()
new=pd.read_csv(NEW,usecols=["frame","fish_id","X","Y","Z","is_interpolated","source"]); log("new loaded",len(new))
old=pd.read_csv(OLD,usecols=["frame","fish_id","X","Y","Z","final_fish_id"]); log("old loaded",len(old))
newp=new.dropna(subset=["X","Y","Z"]); oldp=old.dropna(subset=["X","Y","Z"])
log("NEW frames",int(new.frame.min()),int(new.frame.max()),"nfr",new.frame.nunique(),"present_rows",len(newp),"interp_rows",int(new.is_interpolated.sum()))
log("OLD frames",int(old.frame.min()),int(old.frame.max()),"nfr",old.frame.nunique(),"present_rows",len(oldp))
log("NEW source",new.source.value_counts().to_dict())
# wide arrays indexed by frame for ID_1,2,3
def wide(df):
    p=df.pivot_table(index="frame",columns="fish_id",values=["X","Y","Z"])
    return p
wn=wide(newp); wo=wide(oldp)
labels=["ID_1","ID_2","ID_3"]
def stack(w):
    F=w.index.to_numpy()
    arr=np.full((len(F),3,3),np.nan)
    for li,l in enumerate(labels):
        for ci,c in enumerate(["X","Y","Z"]):
            if (c,l) in w.columns: arr[:,li,ci]=w[(c,l)].to_numpy()
    return F,arr
Fn,An=stack(wn); Fo,Ao=stack(wo)
new_all3=~np.isnan(An).any(axis=(1,2)); old_all3=~np.isnan(Ao).any(axis=(1,2))
log("NEW frames all3",int(new_all3.sum()),"OLD frames all3",int(old_all3.sum()))
mn=dict(zip(Fn[new_all3], An[new_all3])); mo=dict(zip(Fo[old_all3], Ao[old_all3]))
common=np.array(sorted(set(mn)&set(mo)))
log("common all3 frames",len(common))
import itertools
perms=list(itertools.permutations(range(3)))
N=np.stack([mn[f] for f in common]); O=np.stack([mo[f] for f in common])  # (K,3,3)
best_cost=np.full(len(common),np.inf); best_p=np.zeros(len(common),dtype=int); best_d=None
for pi,p in enumerate(perms):
    d=np.linalg.norm(O-N[:,list(p),:],axis=2)  # (K,3)
    c=d.sum(axis=1)
    m=c<best_cost
    best_cost[m]=c[m]; best_p[m]=pi
# recompute matched dist for chosen perm
pid=np.array(perms)[best_p]  # (K,3)
Nm=np.take_along_axis(N,pid[:,:,None].repeat(3,2),axis=1)
dmatch=np.linalg.norm(O-Nm,axis=2)  # (K,3)
rmse=np.sqrt((dmatch**2).mean(axis=1))
is_id=np.array([perms[bp]==(0,1,2) for bp in best_p])
# interfish min in NEW
def interfmin(A):
    return np.minimum.reduce([np.linalg.norm(A[:,0]-A[:,1],axis=1),np.linalg.norm(A[:,0]-A[:,2],axis=1),np.linalg.norm(A[:,1]-A[:,2],axis=1)])
ifm=interfmin(N)
log("identity-perm frac %.4f"%(is_id.mean()),"n_identity",int(is_id.sum()),"n_disagree",int((~is_id).sum()))
log("RMSE mm mean %.3f median %.3f p95 %.3f max %.3f"%(rmse.mean(),np.median(rmse),np.percentile(rmse,95),rmse.max()))
permcounts=pd.Series([str(perms[bp]) for bp in best_p]).value_counts().to_dict()
log("perm counts",permcounts)
# rebind ambiguity: nearest-slot OLD->NEW margin
margins=[]
for i in range(3):
    dd=np.linalg.norm(N-O[:,i:i+1,:],axis=2)  # (K,3) old slot i to all new slots
    ds=np.sort(dd,axis=1)
    margins.append(ds[:,1]-ds[:,0])
margins=np.concatenate(margins)
log("rebind nearest-slot ambiguity(<5mm margin) %.4f"%((margins<5).mean()),"median_margin %.2f"%np.median(margins),"p5 %.2f"%np.percentile(margins,5))
# disagreement timeline ranges
dis=common[~is_id]
rdf_rows=[]
if len(dis):
    splits=np.where(np.diff(dis)>50)[0]
    groups=np.split(np.arange(len(dis)),splits+1)
    idxmap={f:k for k,f in enumerate(common)}
    for g in groups:
        frs=dis[g]; ks=[idxmap[f] for f in frs]
        rdf_rows.append((int(frs.min()),int(frs.max()),len(frs),float(rmse[ks].mean()),"non_identity",float(ifm[ks].min())))
rdf=pd.DataFrame(rdf_rows,columns=["frame_range_start","end","n_frames","rmse_mm","perm_type","min_interfish_mm"]).sort_values("n_frames",ascending=False)
rdf.to_csv(OUT,index=False)
log("wrote",OUT,"n_ranges",len(rdf))
log(rdf.head(12).to_string())
if len(rdf): log("frac disagree-ranges close-contact(<8mm) %.3f"%((rdf.min_interfish_mm<8).mean()),"top10 min ifm",rdf.head(10).min_interfish_mm.round(2).tolist())
