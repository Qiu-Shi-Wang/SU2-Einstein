import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import time
from flint import arb, arb_mat, ctx, arb_poly

ctx.dps=2000
#We will use the arb_poly class.

def rescale(A, rho):
    arho = arb(str(rho))
    initlist = A.coeffs()
    for j in range(len(initlist)):
        initlist[j]*=arho**arb(str(j))
    return arb_poly(initlist)

def eucldiv(A, B): #returns the polynomials Q, R, where A = B*Q + R
    q=arb_poly(arb('0'))
    r=A
    d=B.degree()
    C=B.coeffs()[-1]
    while r.degree()>=d:
        if r.coeffs()[-1].rad()>arb('10')**arb('-50'):
            print(r.coeffs()[-1], r.degree())
        oldrdeg=r.degree()
        s=r.coeffs()[-1]/C
        expo=r.degree()-d
        slist=[arb('0')]*(expo+1)
        slist[-1]+=s
        spoly=arb_poly(slist)
        q+=spoly
        r-=spoly*B
        if r.degree()!=oldrdeg:
            continue
        elif r.degree()==oldrdeg:
            rlast=r.coeffs()[-1]
            if rlast.lower()<=0 and rlast.upper()>=0:
                newr=arb_poly(r.coeffs()[:-1])
                r=newr
            else:
                raise Exception('Euclidean division error!')
    return q, r
    
def signvars(polyseq, x):
    vals=[pol(x) for pol in polyseq]
    valfiltered=[v for v in vals if not v.is_zero()]
    if len(valfiltered)==1:
        return 0
    signvars=0
    ind=0
    i=0
    while i<len(valfiltered)-1:
        if i==0:
            a=valfiltered[i]
            b=valfiltered[i+1]
        elif i!=0:
            a=b
            b=valfiltered[i+1]
        if a*b>0:
            pass
        elif a*b<0:
            signvars+=1
        i+=1
    return signvars
    
def sturmvars(polyraw, araw, braw, rho):
    if rho!=1:
        poly = rescale(polyraw, rho)
        a = araw/arb(str(rho))
        b = braw/arb(str(rho))
    elif rho==1:
        pass
    sturmchain=[]
    sturmchain.append(poly)
    sturmchain.append(poly.derivative())
    while sturmchain[-1].degree()>0:
        newterm=-eucldiv(sturmchain[-2], sturmchain[-1])[1]
        sturmchain.append(newterm)
    Va=signvars(sturmchain, a)
    Vb=signvars(sturmchain, b)
    return Va-Vb

