import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy.stats import scoreatpercentile as sap
from scipy.special import erfinv
from scipy import stats
import pickle,os
from sys import stdout
from PIL import ImageFont, ImageDraw, Image


__all__ = ['getColors','errorbar','pystanErrorbar',
           'printCI','formatAxes',
           'figure','subplot','subplotAnnotate',
           'hist','histCI','plotCIttest1','plotCIttest2',
           'ndarray2latextable','ndarray2gif','plotGifGrid',
           'str2img','plothistCI']
CLR=(0.2, 0.5, 0.6)
# size of figure columns
FIGCOL=[3.27,4.86,6.83] # plosone
FIGCOL=[3.3,5, 7.1] # frontiers
FIGCOL=[3.3,5,7.1] # plosone new
FIGCOL=[2.87,4.3,5.74,9]# peerj


SMALL=6
MEDIUM=9
plt.rc('axes', titlesize=MEDIUM)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)    # fontsize of the tick labels

# TODO custom ppl style histogram
def getColors(N):
    ''' creates set of colors for plotting

        >>> len(getColors(5))
        5
        >>> getColors(5)[4]
        (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
        >>> N=14
        >>> c=getColors(N)
        >>> plt.scatter(range(N),range(N),color=c)
        >>> plt.show()
    '''
    clrs=[]
    cm = plt.get_cmap('Paired')
    for i in range(N+1):
        clrs.append(cm(1.*i/float(N))) 
        #plt.plot(i,i,'x',color=clrs[i])
    clrs.pop(-2)
    return clrs

def imshow(*args,**kwargs):
    plt.imshow(*args,**kwargs)

#plt.ion()
#imshow(np.array([[1,2,3],[2,1,2],[3,1,2]]))

def formatAxes(ax):
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False,axis='x')
    ax.grid(True,axis='y')
    
def figure(*args,**kwargs):
    ''' wrapper around matplotlib.figure
        additionally supports following kwargs
        size - 1,2 or 3 respectively for small, medium, large width
        aspect - [0,inf] height to width ratio 
    '''
    if not 'figsize' in kwargs:
        if 'size' in kwargs: w= FIGCOL[kwargs.pop('size')-1]
        else: w=FIGCOL[0]
        if 'aspect' in kwargs: h=kwargs.pop('aspect')*w
        else: h=w
        kwargs['figsize']=(w,h)
    fig=plt.figure(*args,**kwargs)
    formatAxes(plt.gca())
    return fig

def hist(*args,**kwargs):
    '''
        >>> dat=np.random.randn(1000)*10+5
        >>> x=np.linspace(-30,30,60)
        >>> hist(dat,bins=x)
    '''
    if not 'facecolor' in kwargs: kwargs['facecolor']=CLR
    if not 'edgecolor' in kwargs: kwargs['edgecolor']='w'
    plt.hist(*args,**kwargs)

def histCI(*args,**kwargs):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> histCI(x,bins=bn)
    '''
    if 'plot' in kwargs: plot=kwargs.pop('plot')
    else: plot=True
    if 'alpha' in kwargs: alpha=kwargs.pop('alpha')
    else: alpha=0.05
    a,b=np.histogram(*args,**kwargs)
    m=b.size
    n=args[0].shape[0]
    c=stats.norm.ppf(alpha/(2.*m))/2.*(m/float(n))**0.5
    l=np.square(np.maximum(np.sqrt(a)-c,0))
    u=np.square(np.sqrt(a)+c)
    if plot: plothistCI(a,b,l,u)
    return a,b,l,u
def symhist(x1,x2,bins):
    ''' symmetric histogram of two data sets
        >>> symhist(np.random.randn(100),np.random.randn(100)+1,np.linspace(-3,4,15))
        >>> plt.show()
    '''
    bw=bins[1]-bins[0]
    a1=np.histogram(x1,bins=bins,normed=True)
    plt.barh(bins[:-1],-a1[0],ec='w',fc='y',height=bw,lw=0.1)
    a2=np.histogram(x2,bins=bins,normed=1)
    plt.barh(bins[:-1],a2[0],ec='w',fc='y',height=bw,lw=0.1)
    xmax=max(plt.xlim())
    plt.xlim([-xmax,xmax])
    plt.ylim([bins[0],bins[-1]])
    ax=plt.gca()
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])


def plothistCI(a,b,l,u):
    '''
        >>> x=np.random.randn(1000)
        >>> bn=np.linspace(-3,3,41)
        >>> a,b,l,u=histCI(x,bins=bn)
        >>> plothistCI(a,b,l,u)
    '''
    b=b[:-1]+np.diff(b)/2.
    plt.plot(b,a)
    x=np.concatenate([b,b[::-1]])
    ci=np.concatenate([u,l[::-1]])
    plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                alpha=0.2,fill=True,fc='red',ec='red'))

def kernelreg(x,y,xnew,ciwidth=0.5,Kwidth=1):
    '''
        Nadaray-Watson kernel estimator 
        from Wasserman(2004) chap 20.4
        x - predictor NxK1x...xKd ndarray
        y - outcome, N-length ndarray
        xnew - predictor for prediction MxK1x...xKd ndarray
        returns list of
            estimated outcome for xnew, M-length ndarray
            lower confidence interval
            upper confidence interval
            leave-one-out crossvalidation score
        example:
        >>> x=np.linspace(-10,10,200)
        >>> x+=np.random.randn(x.size)*0.1
        >>> x=np.sort(x)
        >>> y=np.sin(x)+np.random.randn(x.size)
        >>> xnew=np.linspace(-10,10,1000)
        >>> for k in [0.05,0.5,1,2,5,10]:
        >>>     plt.figure()
        >>>     plt.plot(x,y,'.')
        >>>     ynew,lcf,ucf,J=kernelreg(x,y,xnew,Kwidth=k)
        >>>     plt.plot(xnew,ynew)
        >>>     plt.plot(xnew,lcf,'r')
        >>>     plt.plot(xnew,ucf,'r')
        >>>     plt.title('h=%.01f, J=%.01f'%(k,J))
    '''
    K=lambda x: (2*np.pi)**-0.5*np.exp(-np.square(x)/2.)
    xlist=x.tolist();N=len(xlist)
    xnlist=xnew.tolist();M=len(xnlist)
    ynew=np.zeros(M)*np.nan
    w=np.zeros((M,N))
    for i1 in range(M):
        for i2 in range(N):
            d=np.linalg.norm(xnlist[i1]-xlist[i2])
            w[i1,i2]=K((d)/float(Kwidth))
    for i1 in range(M):
        ynew[i1]=w[i1,:].dot(y)/w[i1,:].sum()
    sigma=np.square(np.diff(y))
    sigma= (sigma.sum()/float(2*sigma.size))**0.5
    se=sigma*np.sqrt(np.square(w).sum(1))
    rng=0
    w=np.zeros((N,N))
    r=np.zeros(N)*np.nan
    for i1 in range(N):
        for i2 in range(N):
            d=np.linalg.norm(xnlist[i1]-xlist[i2])
            if d>rng: rng=d
            w[i1,i2]=K((d)/float(Kwidth))
    for i1 in range(N):
        r[i1]=w[i1,:].dot(y)/w[i1,:].sum()
    J=np.square((y-r)/(1-K(0)/w.sum(axis=1))).sum()
    q=erfinv(0.5*(1+ciwidth**(3*Kwidth/rng)))
    return ynew, ynew-q*se, ynew+q*se,J

def subplot(*args):
    ax=plt.subplot(*args)
    formatAxes(ax)
    return ax

def subplotAnnotate(loc='nw',nr=None,clr='k'):
    if type(loc) is list and len(loc)==2: ofs=loc
    elif loc=='nw': ofs=[0.1,0.9]
    elif loc=='sw': ofs=[0.1,0.1]
    elif loc=='se': ofs=[0.9,0.1]
    elif loc=='ne': ofs=[0.9,0.9]
    else: raise ValueError('loc only supports values nw, sw, se and ne')
    ax=plt.gca()
    if nr is None: nr=ax.colNum*ax.numRows+ax.rowNum
    plt.text(plt.xlim()[0]+ofs[0]*(plt.xlim()[1]-plt.xlim()[0]),
            plt.ylim()[0]+ofs[1]*(plt.ylim()[1]-plt.ylim()[0]), 
            str(chr(65+nr)),horizontalalignment='center',verticalalignment='center',
            fontdict={'weight':'bold'},fontsize=12,color=clr)

def _errorbar(out,x,clr='k'):
    plt.plot([x,x],out[1:3],color=clr)
    plt.plot([x,x],out[3:5],
        color=clr,lw=3,solid_capstyle='round')
    plt.plot([x],[out[0]],mfc=clr,mec=clr,ms=8,marker='_',mew=2)

def _horebar(d,xs,clr):
    ''' code snippet for horizontal errorbar'''
    for i in range(d.shape[1]):
        x=xs[i]
        plt.plot([sap(d[:,i],2.5),sap(d[:,i],97.5)],[x,x],color=clr)
        plt.plot([sap(d[:,i],25),sap(d[:,i],75)],[x,x],
            color=clr,lw=3,solid_capstyle='round')
        plt.plot([np.median(d[:,i])],[x],mfc=clr,mec=clr,ms=8,marker='|',mew=2)
    plt.gca().set_yticks(xs)

def plotCIttest1(y,x=0,alpha=0.05,clr=CLR):
    ''' single group t-test'''
    m=y.mean();df=y.size-1
    se=y.std()/y.size**0.5
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr)
    return out
    
def plotCIttest2(y1,y2,x=0,alpha=0.05,clr='k'):
    n1=float(y1.size);n2=float(y2.size);
    v1=y1.var();v2=y2.var()
    m=y2.mean()-y1.mean()
    s12=(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))**0.5
    se=s12*(1/n1+1/n2)**0.5
    df= (v1/n1+v2/n2)**2 / ( (v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)) 
    cil=stats.t.ppf(alpha/2.,df)*se
    cii=stats.t.ppf(0.25,df)*se
    out=[m,m-cil,m+cil,m-cii,m+cii]
    _errorbar(out,x=x,clr=clr)
    return out
    
def errorbar(y,clr=CLR,x=None,labels=None,plot=True):
    ''' customized error bars
        y - NxM ndarray containing results of
            N simulations of M random variables
        x - array with M elements, position of the bars on x axis 
        clr - bar color
        labels - array with xtickslabels

        >>> errorbar(np.random.randn(1000,10)+1.96)    
    '''
    out=[]
    d=np.array(y);
    if d.ndim<2: d=np.array(y,ndmin=2).T
    if not x is None: x=np.array(x)
    if x is None or x.size==0: x=np.arange(d.shape[1])
    elif x.size==1: x=np.ones(d.shape[1])*x[0]
    elif x.ndim!=1 or x.shape[0]!=d.shape[1]:
        x=np.arange(0,d.shape[1])
    ax=plt.gca()
    for i in range(d.shape[1]):
        out.append([np.median(d[:,i]),sap(d[:,i],2.5),sap(d[:,i],97.5),
                    sap(d[:,i],25),sap(d[:,i],75)])
        if plot: _errorbar(out[-1],x=x[i],clr=clr)
    if plot:
        ax.set_xticks(x)
        if not labels is None: ax.set_xticklabels(labels)
        plt.xlim([np.floor(np.min(x)-1),np.ceil(np.max(x)+1)])
    return np.array(out)

def pystanErrorbar(w,keys=None):
    """ plots errorbars for variables in fit
        fit - dictionary with data extracted from Pystan.StanFit instance 
    """
    kk=0
    ss=[];sls=[]
    if keys is None: keys=list(w.keys())[:-1]
    for k in keys:
        d= w[k]
        if d.ndim==1:
            ss.append(d);sls.append(k)
            continue
            #d=np.array(d,ndmin=2).T
        d=np.atleast_3d(d)
        for h in range(d.shape[2]):
            kk+=1; figure(num=kk)
            #ppl.boxplot(plt.gca(),d[:,:,h],sym='')
            errorbar(d[:,:,h])
            plt.title(k)
    #ss=np.array(ss)
    for i in range(len(ss)):
        print(sls[i], ss[i].mean(), 'CI [%.3f,%.3f]'%(sap(ss[i],2.5),sap(ss[i],97.5)))
def printCI(w,var=None,decimals=3):
    sfmt=' {:.{:d}f} [{:.{:d}f},{:.{:d}f}]'
    def _print(b):
        d=np.round([np.median(b), sap(b,2.5),sap(b,97.5)],decimals).tolist()
        print(sfmt.format(d[0],decimals,d[1],decimals,d[2],decimals))
        #print var+' %.3f, CI %.3f, %.3f'%tuple(d) 
    if var is None: d=w;var='var'
    else: d=w[var]
    if d.ndim==2:
        for i in range(d.shape[1]):
            _print(d[:,i])
    elif d.ndim==1: _print(d)
    
def saveStanModel(sm,fname='test'):
    if fname.count(os.path.sep): path=''
    else: path = os.getcwd()+os.path.sep+'standata'+os.path.sep
    try: os.mkdir(path)
    except: OSError
    with open(f'{path}{fname}.sm', 'wb') as f: pickle.dump(sm, f)   
    
def loadStanModel(fname='test'):  
    if fname.count(os.path.sep): path=''
    else: path = os.getcwd()+os.path.sep+'standata'+os.path.sep  
    with open(f'{path}{fname}.sm', 'rb') as f: return pickle.load(f)
           
def saveStanFit(fit,fname='test'):     
    if fname.count(os.path.sep): path=''
    else: path = os.getcwd()+os.path.sep+'standata'+os.path.sep
    try: os.mkdir(path)
    except: OSError
    w=fit.extract()
    f=open(path+fname+'.stanfit','wb')
    pickle.dump(w,f)
    f.close()
    f=open(path+fname+'.check','w')
    f.write(str(fit))
    f.close()
def loadStanFit(fname):
    if fname.count(os.path.sep): path=''
    else: path = os.getcwd()+os.path.sep+'standata'+os.path.sep
    f=open(path+fname+'.stanfit','rb')
    out=pickle.load(f)
    f.close()
    return out

def ndarray2latextable(array,decim=2,hline=[0],vline=None,nl=1):
    ''' array - 2D numpy.ndarray with shape (rows,columns)
        decim - decimal precision of float, use 0 for ints
            should be int scalar or a list of list,len(decim)=nr cols
    '''
    ecol=' \\\\\n';shp=array.shape
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|'
    if vline is None: vline=range(shp[1])
    for i in range(shp[1]):
        out+=['c','l'][int(i<nl)]+['','|'][int(i in set(vline))]
    out=out+'|}\n\\hline\n'
    for i in range(shp[0]):
        for j in range(shp[1]):
            if type(decim) is list: dc=decim[j]
            else: dc=decim
            if type(array[i,j])==np.str_ or type(array[i,j])==str: out+='%s'%array[i,j]
            elif dc==0: out+='%d'%int(array[i,j])
            else:
                flt='{: .%df}'%dc
                out+=flt.format(np.round(array[i,j],dc))
            if j<shp[1]-1: out+=' & '
        out+=ecol
        if hline.count(i)==1: out+='\\hline\n'
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print(out)
    
def ndsamples2latextable(data,decim=2):
    ''' data - 3D numpy.ndarray with shape (rows,columns,samples)'''
    elem='{: .%df} [{: .%df}, {: .%df}]'%(decim,decim,decim)
    ecol=' \\\\\n'
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|l|'+data.shape[1]*'c|'+'}\n\\hline\n'
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out+=elem.format(data[i,j,:].mean(),
                stats.scoreatpercentile(data[i,j,:],2.5),
                stats.scoreatpercentile(data[i,j,:],97.5))
            if j<data.shape[1]-1: out+=' & '
        out+=ecol
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print(out)


def ndarray2gif(path,array,duration=0.1,addblank=False,
                plottime=False,snapshot=1,UC=255,LC=20):
    '''
    path - file path, including filename and suffix,
        supports .gif, .avi (lossless h264 codec)
    array - 3d numpy array, zeroth dim is the time axis,
            dtype uint8 or float in [0,1]
    duration - frame duration
    snapshot - 0=first frame, 1=midframe, 2=last frame

    Example:
    
    >>> im = np.zeros((200,200), dtype=np.uint8)
    >>> im[10:30,:] = 100
    >>> im[:,80:120] = 255
    >>> im[-50:-40,:] = 50
    >>> images = [im*1.0, im*0.8, im*0.6, im*0.4, im*0]
    >>> images = [im, np.rot90(im,1), np.rot90(im,2), np.rot90(im,3), im*0]
    >>> ndarray2gif('test',np.array(images), duration=0.5) 
    '''
    path,suf=path.rsplit('.')
    if array.dtype.type is np.float64 or array.dtype.type is np.float32:
        array=np.uint8(255*array)
    if plottime:
        shp=list(array.shape); shp[1]+=50;
        T=np.ones(shp,dtype=np.uint8)*UC;T[:,:-50,:]=array
        for f in [-1,0,1]:
            T[:,-40:-10,shp[2]/10+f]=LC
            T[:,-25+f,shp[2]/10:9*shp[2]/10]=LC
            T[:,-40:-10,9*shp[2]/10+f]=LC
            for ff in [3,5,7]: T[:,-35:-15,ff*shp[2]/10+f]=LC
            for t in range(T.shape[0]):
                T[t,-40:-10,int((1+float(8*t)/T.shape[0])*shp[2]/10.)+f]=LC
        array=T
    if addblank:
        temp=np.zeros((array.shape[0]+1,array.shape[1],array.shape[2]),dtype=np.uint8)
        temp[1:,:,:]=array
        array=tem
    if array.shape[1]%2==1: array=array[:,:-1,:]
    if array.shape[2]%2==1: array=array[:,:,:-1]
    for k in range(array.shape[0]):
        I=Image.fromarray(array[k,:,:])
        I.save('temp%04d.png'%k)   
    if suf=='gif': os.system('convert -delay %f temp*.png %s.gif'%(duration,path))
    elif suf=='avi':
        cmd='avconv -r '+str(int(1/duration))+ ' -i temp%04d.png -c:v h264 '+path+'.avi'
        print(cmd)
        os.system(cmd)
    else: raise ValueError
    shp=array.shape[0]
    os.system('cp temp%04d.png %s.png'%([0,shp/2,shp-1][snapshot],path))
    for k in range(array.shape[0]):
        os.system('rm temp%04d.png'%k)

def str2img(inp,size,fontpath="/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"):
    image= Image.fromarray(np.zeros((size,1+int(np.round(size*2/3.*len(inp))))))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontpath,size)
    draw.text((0,0), inp, font=font)
    return np.asarray(image)

def plotGifGrid(dat,fn='test',bcgclr=0,forclr=None,text=[],duration=0.1,
                plottime=False,snapshot=False,F=68,P=64,tpG=False,tpL=False):
    '''
        text = [[text1,textsize1,posx1,posy1],[text2, ...
    '''
    if forclr==None: forclr=1-bcgclr
    if forclr==bcgclr: forclr=1
    offset=8 # nr pixels for border padding
    if tpG: cols=len(dat); rows=len(dat[0])
    else: rows=len(dat); cols=len(dat[0])
    h=P;w=P;t=F
    R=bcgclr*np.ones((t+1,(h+offset)*rows,(w+offset)*cols),
                     dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            i=((offset+h)*row+offset/2,(offset+w)*col+offset/2)
            if tpG: temp=dat[col][row]
            else: temp=dat[row][col]
            #temp=np.rollaxis(dat[row][col].T,2)
            for tt in range(F):
                if temp.shape[0]==F: tempp=temp[tt,:,:]
                elif temp.shape[1]==F: tempp=temp[:,tt,:]
                elif temp.shape[2]==F: tempp=temp[:,:,tt]
                else: raise ValueError
                if tpL: tempp=tempp.T
                R[1+tt,i[0]:i[0]+h,i[1]:i[1]+w]=tempp
                
    mnxy=[0,0]
    for t in text:
        t.append(str2img(t[0],t[1]))
        shp=t[-1].shape
        if mnxy[0]>t[2]-shp[0]/2-offset:mnxy[0]=t[2]-shp[0]/2-offset
        if mnxy[1]>t[3]-shp[1]/2-offset:mnxy[1]=t[3]-shp[1]/2-offset
    if np.any(np.array(mnxy)<0):
        shp=list(R.shape); shp[1]-=mnxy[0];shp[2]-=mnxy[1]
        T=np.ones(shp,dtype=np.float32)*bcgclr
        T[:,-mnxy[0]:,-mnxy[1]:]=np.copy(R)
        R=T
    for t in text:
        px,py=t[-1].nonzero()
        R[:,px+t[2]+t[-1].shape[0]/2+offset,
          py+t[3]+t[-1].shape[1]/2+offset]=forclr
    ndarray2gif(fn,np.uint8(R*255),duration=duration,
                plottime=plottime,snapshot=snapshot,
                UC=np.uint8(bcgclr*255),LC=np.uint8(255-bcgclr*255))
def printProgress(iteration, total, time,prefix='', decimals=1, bar_length=30):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = 'O' * filled_length + '-' * (bar_length - filled_length)
    if iteration>0:
        suffix=" ETA "+str_format.format(time*(total/iteration-1)/60.)+ 'mins'
    else: suffix=''
    stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))
    if iteration == total:
        stdout.write('\n')
    stdout.flush()  

