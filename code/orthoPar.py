import pystan
from matusplotlib import saveStanFit,loadStanFit,saveStanModel,loadStanModel,pystanErrorbar
from matusplotlib import loadStanFit,errorbar,subplotAnnotate
from scipy.stats import scoreatpercentile as sap
import numpy as np
import pylab as plt
SEED=6
N=10;M=50
PARTRUEGAM={'N':N,'M':M,'g_m':2,'g_s':0.5,'t_m':1,'t_s':1}
PARTRUEWEI={'N':N,'M':M,'g_m':5,'g_s':2,'t_m':-1,'t_s':1}

def runStan(compile=False):
    invdigamma='''
    functions{
        real invdigamma(real x){
            real y; real L;
            if (x>=-5.24) y=1/log(1+exp(-x));
            else y=1/(digamma(1)-x);
            L=digamma(y)-x;
            while (fabs(L)>1e-12){
                y=y-L ./trigamma(y);
                L=digamma(y)-x;}
            return y;}} '''
    gamma='''
    data{
        int<lower=0> N;
        int<lower=0> M;
        real<lower=0> y[N,M];}
    parameters{
        real g[N];
        real t[N];
        real g_m;real t_m;
        real<lower=0> g_s;
        real<lower=0> t_s;}
    model{
        for (n in 1:N){
            g[n]~normal(g_m,g_s);
            t[n]~normal(t_m,t_s);
            for (m in 1:M){
                y[n,m]~gamma(invdigamma(g[n]-t[n]),exp(-t[n]));}}}
    '''

    gammaGen = """
    data {
        int<lower=0> N; 
        int<lower=0> M;
        real g_m;real t_m;
        real<lower=0> g_s;
        real<lower=0> t_s;
    }generated quantities{
        real g[N];real t[N];
        real<lower=0> y[N,M];
        for (n in 1:N){
            g[n]=normal_rng(g_m,g_s);
            t[n]=normal_rng(t_m,t_s);
            for (m in 1:M){
                y[n,m]=gamma_rng(invdigamma(g[n]-t[n]),exp(-t[n]));
    }}}
    """
    weibull = """
    data {
        int<lower=0> N;
        int<lower=0> M;
        vector<lower=0>[M] y[N];
    }parameters {
        real<lower=0> g[N];
        real t[N];
        real g_m;real t_m;
        real<lower=0> g_s;
        real<lower=0> t_s;
    }model {
        g~normal(g_m,g_s);
        t~normal(t_m,t_s);
        for (n in 1:N)
            y[n]~weibull(4.313501020391736/(g[n]-t[n]),exp(t[n]));
    }"""

    weibullGen= """
    data {
        int<lower=0> N;
        int<lower=0> M;
        real g_m;real t_m;
        real<lower=0> g_s;
        real<lower=0> t_s;
    }generated quantities{
        real<lower=0> y[N,M];
        real g[N];
        real t[N];
        for (n in 1:N){
            g[n]=normal_rng(g_m,g_s);
            t[n]=normal_rng(t_m,t_s);
            for (m in 1:M)
                y[n,m]=weibull_rng(4.313501020391736/(g[n]-t[n]),exp(t[n]));
    }}"""
    if compile:
        for m in [['gamma',invdigamma+gamma,invdigamma+gammaGen],
            ['weibull',weibull,weibullGen]]:
            for i in range(2):
                sm=pystan.StanModel(model_code=m[1+i])
                saveStanModel(sm,m[0]+['','Gen'][i])     
    for i in range(2):
        nm=['gamma','weibull'][i]
        sm=loadStanModel(nm+'Gen')
        fit=sm.sampling(data=[PARTRUEGAM,PARTRUEWEI][i],
            chains=4,n_jobs=4,seed=SEED,thin=1,iter=30,
            warmup=0,algorithm="Fixed_param")
        saveStanFit(fit,nm+'Gen')
        w=fit.extract()
        y=w['y'][0,:,:]
        sm=loadStanModel(nm)
        fit=sm.sampling(data={'N':N,'M':M,'y':y},chains=4,
            n_jobs=4,seed=SEED,thin=2,iter=2000,warmup=1000)   
        saveStanFit(fit,nm)

def plotSF():
    plt.figure(dpi=400,figsize=(7,7));h=1
    lbls=[]
    for i in range(N): lbls.append(f'$\\gamma_{i}$')
    for i in range(N): lbls.append(f'$\\tau_{i}$')
    lbls.extend(['$\\mu_\\gamma$','$\\mu_\\tau$',
        '$\\sigma_\\gamma$','$\\sigma_\\tau$'])
    for mdl,part in [['gamma',PARTRUEGAM],['weibull',PARTRUEWEI]]:
        plt.subplot(2,1,h);h+=1
        w=loadStanFit(mdl+'Gen')
        for k in ['g','t']:part[k]=w[k][0,:]
        w=loadStanFit(mdl)
        kk=0
        est=[];tval=[]
        keys=list(w.keys())[:-1]
        for k in keys:
            d= w[k]
            if d.ndim==1:
                d=d[:,np.newaxis]
                tval.append(part[k])
            else: 
                tval.extend(list(part[k]))
            est.append(d)
        est=np.concatenate(est,axis=1)
        
        errorbar(est,labels=lbls,clr='grey');
        plt.grid(True,axis='y')
        plt.plot(np.arange(len(tval))+0.3,tval,'+k')
        plt.xlabel('Parameter')
        plt.ylabel('Parameter value')
        subplotAnnotate(loc='ne',fs=20)
    plt.savefig('stan.png',bbox_inches='tight')

if __name__=='__main__':
    runStan(compile=True)
    plotSF()



