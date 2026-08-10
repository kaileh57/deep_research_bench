#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.environ.get('AICROWD_RESEARCH','/tmp/aicrowd'))
from whestfloor.cumulants import kappa3_star, kappa3_tree
from whestfloor.official_seeds import make_official_mlp
from whestfloor.parquet_lite import read_columns_over_http
W,D=256,32
RANKS=(8,16,32,64)
EPS=1e-30

def k3(m,m2,m3): return m3-3*m*m2+2*m**3

def c21(m,M11,M21,m2):
    return M21-2*m[:,None]*M11-m2[:,None]*m[None,:]+2*m[:,None]**2*m[None,:]

def blocks(A,k,c):
    A2=A*A; diag=np.sum(A2*A*k[:,None],axis=0)
    allpair=np.sum(A2*(c@A),axis=0)
    known=3*allpair-2*diag
    return known,diag,known-diag

def tri(U,Q):
    C=U@U.T; d=np.diag(C); B=C*C
    G=np.einsum('ir,ia,is->ars',U,Q,U,optimize=True); G2=G@G
    s0=np.einsum('aij,aji->a',G2,G,optimize=True)
    BQ=B@Q
    return s0-3*np.sum(Q*Q*d[:,None]*BQ,axis=0)+2*np.sum(Q**3*d[:,None]**3,axis=0)

def factors(C):
    e,V=np.linalg.eigh((C+C.T)*.5); e=np.maximum(e,0); o=np.argsort(e)[::-1]; e=e[o]; V=V[:,o]
    out={}; tot=max(e.sum(),EPS)
    for r in RANKS: out[r]=(V[:,:r]*np.sqrt(e[:r])[None,:],float(e[:r].sum()/tot))
    return out,float(tot*tot/max(np.sum(e*e),EPS))

def seeds(n):
    a=[]
    for s in range(28):
        u=f'https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v1-phase1/data/full-{s:05d}-of-00028.parquet'
        _,c=read_columns_over_http(u,['mlp_seed']); a.extend(np.asarray(c['mlp_seed'],dtype=np.int64).tolist())
        if len(a)>=n: break
    return np.asarray(a[:n],dtype=np.int64)

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-30 and np.std(y)>1e-30 else float('nan')

def stat(name,p,y):
    q=float(np.mean((p-y)**2)); z=float(np.mean(y*y))
    return {'name':name,'mse':q,'rms':math.sqrt(q),'r2_zero':1-q/max(z,EPS),'corr':corr(p,y)}

def beta(x,y): return float(x@y/(x@x+1e-30))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=12); ap.add_argument('--train',type=int,default=9); ap.add_argument('--out',default='k3-cycle-out')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ss=seeds(a.n); cache='/tmp/hf-k3-cache'; Path(cache).mkdir(exist_ok=True)
    cols={k:[] for k in ['net','layer','target','known','diag','pair','bulk','star','tree','tri8','tri16','tri32','tri64']}
    specs=[]; t0=time.time()
    for ni in range(a.n):
        print(f'NET {ni}/{a.n} seed={ss[ni]}',flush=True)
        p=hf_hub_download('keenanpepper/arc-whestbench-higher-moments-2026',f'full/mlp_{ni:05d}.npz',repo_type='dataset',cache_dir=cache)
        ww=[np.asarray(x,dtype=np.float64) for x in make_official_mlp(W,D,int(ss[ni]))]
        with np.load(p,allow_pickle=False) as z:
            assert int(z['global_index'])==ni
            hm=np.asarray(z['mean'],dtype=np.float64); h11=np.asarray(z['M11'],dtype=np.float64); h21=np.asarray(z['M21'],dtype=np.float64)
            h2=np.asarray(z['m2'],dtype=np.float64); h3=np.asarray(z['m3'],dtype=np.float64)
            zm=np.asarray(z['pre_mean'],dtype=np.float64); z11=np.asarray(z['pre_M11'],dtype=np.float64); z2=np.asarray(z['pre_m2'],dtype=np.float64); z3=np.asarray(z['pre_m3'],dtype=np.float64)
            for l in range(31):
                A=ww[l+1]; kd=k3(hm[l],h2[l],h3[l]); cc=c21(hm[l],h11[l],h21[l],h2[l]); known,di,pa=blocks(A,kd,cc)
                target=k3(zm[l+1],z2[l+1],z3[l+1]); bulk=target-known
                C=(z11[l]-np.outer(zm[l],zm[l])); C=(C+C.T)*.5; s=np.sqrt(np.maximum(np.diag(C),1e-14)); al=zm[l]/s
                ph=np.exp(-.5*al*al)/math.sqrt(2*math.pi); R=np.clip(C/np.outer(s,s),-.999999,.999999)
                star=kappa3_star(A,zm[l],s,R,umax=3); tree=kappa3_tree(A,zm[l],s,R,K2=8,T3=3)
                fs,pr=factors(C); Q=(ph/s)[:,None]*A; tv={r:tri(fs[r][0],Q) for r in RANKS}
                for k,v in [('net',np.full(W,ni)),('layer',np.full(W,l)),('target',target),('known',known),('diag',di),('pair',pa),('bulk',bulk),('star',star),('tree',tree)]: cols[k].append(v)
                for r in RANKS: cols[f'tri{r}'].append(tv[r])
                specs.append([ni,l,pr]+[fs[r][1] for r in RANKS])
                if l in (0,1,3,7,15,23,30): print(f' L{l+1:02d} target={np.sqrt(np.mean(target**2)):.3e} tree={np.sqrt(np.mean((tree-target)**2)):.3e} +c64={np.sqrt(np.mean((tree+tv[64]-target)**2)):.3e}',flush=True)
        print(' elapsed',round(time.time()-t0,1),flush=True)
    x={k:np.concatenate(v).astype(np.float32) for k,v in cols.items()}; np.savez_compressed(out/'rows.npz',**x)
    tr=x['net']<a.train; te=~tr; y=x['target'][te].astype(np.float64); yt=x['target'][tr].astype(np.float64); tree=x['tree'][te].astype(np.float64); treet=x['tree'][tr].astype(np.float64)
    res={'n':a.n,'train':a.train,'runtime_s':time.time()-t0,'bulk_energy_fraction':float(np.mean(x['bulk'][te]**2)/np.mean(y*y)),'predictions':[stat('known_true_blocks',x['known'][te],y),stat('star_true_state',x['star'][te],y),stat('tree_true_state',tree,y)],'cycle':{},'per_layer':[]}
    for r in RANKS:
        q=x[f'tri{r}'][te].astype(np.float64); qt=x[f'tri{r}'][tr].astype(np.float64)
        b=beta(qt,yt-treet); bb=beta(qt,x['bulk'][tr].astype(np.float64))
        res['cycle'][str(r)]={'beta_tree_residual':b,'beta_bulk':bb,'corr_tree_residual':corr(q,y-tree),'corr_bulk':corr(q,x['bulk'][te])}
        res['predictions'] += [stat(f'tree+unit_cycle_r{r}',tree+q,y),stat(f'tree+fit_cycle_r{r}',tree+b*q,y),stat(f'known+fit_cycle_r{r}',x['known'][te]+bb*q,y)]
    pred=tree.copy(); bl=[]
    for l in range(31):
        lt=tr&(x['layer']==l); le=te&(x['layer']==l); q=x['tri64'][le].astype(np.float64); qtr=x['tri64'][lt].astype(np.float64)
        b=beta(qtr,x['target'][lt].astype(np.float64)-x['tree'][lt].astype(np.float64)); pred[x['layer'][te]==l]+=b*q; bl.append(b)
        yy=x['target'][le].astype(np.float64); tt=x['tree'][le].astype(np.float64)
        res['per_layer'].append({'layer':l,'beta':b,'tree_rms':math.sqrt(np.mean((tt-yy)**2)),'unit_rms':math.sqrt(np.mean((tt+q-yy)**2)),'fit_rms':math.sqrt(np.mean((tt+b*q-yy)**2)),'corr':corr(q,yy-tt)})
    res['layerwise']=stat('tree+layerwise_cycle64',pred,y); res['layerwise']['beta']=bl
    S=np.asarray(specs); res['participation_ratio']=float(S[:,2].mean()); res['variance_fraction']={str(r):float(S[:,3+i].mean()) for i,r in enumerate(RANKS)}
    (out/'summary.json').write_text(json.dumps(res,indent=2,sort_keys=True)); np.savez_compressed(out/'model.npz',beta64=np.asarray(bl))
    print('\nHELDOUT'); [print(v['name'],v['rms'],v['r2_zero'],v['corr']) for v in res['predictions']]; print('layerwise',res['layerwise']['rms'],res['layerwise']['r2_zero'])
if __name__=='__main__': main()
