import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import time
from flint import arb, arb_mat, ctx, arb_poly

from sturm import sturmvars
from estimates_inf import inf_try_from_sol

#this is in good part Buttsworth--Hodgkinson code
from arb_cheby import chebyfit, T_rational, arb_chebyT

ctx.dps = 1000

b1=arb('.1')
h=arb('1.5')

startclock=time.time()

def adjustlen(vec, r):
    while len(vec)<r:
        vec.append(arb('0'))
    return vec

def vectnorm(vec):
    n=0
    for a in vec:
        n+=a*a
    return n.sqrt()

def indexj(j, s1, s2):
    c=arb('0')
    for k in range(j+1):
        c+=s1[k]*s2[j-k]
    return c

#we will store a power series as a list of coefficients [a_0, a_1, ..., a_n].
class series:
    def __init__(self, values):
        self.deg = len(values)
        self.s = values
        
    def __str__(self):
        return str(self.s)
    
    def __neg__(obj1):
        return series([-a for a in obj1.s])
        
    def __add__(obj1, obj2):
        if obj1.__class__ == series and obj2.__class__ == series:
            if obj1.deg<=obj2.deg:
                short=obj1.s
                longu=obj2.s
            else:
                short=obj2.s
                longu=obj1.s
            firstpart=[short[i]+longu[i] for i in range(len(short))]
            totallist=firstpart+longu[len(short):]
            return series(totallist)
        if obj1.__class__ == series and obj2.__class__ != series:
            constval=obj1.s[0]+obj2
            backlist = obj1.s[1:]
            return series([constval]+backlist)
        if obj1.__class__ != series and obj2.__class__ == series:
            return series.__add__(obj2, obj1)
    def __radd__(obj1, obj2):
        return series.__add__(obj1, obj2)
            
    def __sub__(obj1, obj2):
        return series.__add__(obj1,-obj2)
        
    def __rsub__(obj1, obj2):
        return series.__sub__(obj2, obj1)
            
    def __mul__(obj1, obj2):
        if obj1.__class__ == series and obj2.__class__ == series:
            if obj1.deg != obj2.deg:
                maxdeg=max(obj1.deg, obj2.deg)
                return series.__mul__(series(adjustlen(obj1.s,maxdeg)), series(adjustlen(obj2.s, maxdeg)))
            else:
                n=obj1.deg
                result=[]
                s1, s2 = obj1.s, obj2.s
                for j in range(n):
                    result.append(indexj(j, s1, s2))
                return series(result)
        if obj1.__class__ == series and obj2.__class__ != series:
            newlist = [obj2*a for a in obj1.s]
            return series(newlist)
            
    def __rmul__(obj1,obj2):
        return series.__mul__(obj1,obj2)
        
    def inv(s1): #formula taken from https://functions.wolfram.com/GeneralIdentities/7/
        if s1.__class__!=series:
            return arb('1')/s1
            
        if s1.s[0]==0:
            raise Exception('can\'t invert a series vanishing at zero!')
        n=s1.deg-1
        
        slist=s1.s
        b0=slist[0]
        pjlist=[]
        for j in range(n+1):
            plist=[]
            plist.append(1)
            for k in range(1,n+1):
                S=arb('0')
                for m in range(1,k+1):
                    S+=arb(j*m + m - k)*slist[m]*plist[k-m]
                plist.append(1/(b0*k)*S)
            
            pjlist.append(plist)
        
        result=[]
        for k in range(n+1):
            coeff=arb('0')
            for r in range(k+1):
                backstr=str(r+1)
                if r%2==0:
                    prefac=arb('1/'+backstr)
                elif r%2==1:
                    prefac=arb('-1/'+backstr)
                coeff+=prefac * arb(str(math.comb(k, r))) * pjlist[r][k]
            result.append(1/b0 * arb(k+1)*coeff)
        
        return series(result)         
    
    
    def polyseries(expo, term, degree):
        L=[arb('0')]*(degree+1)
        L[expo]+=term
        return series(L)
    
    def addnewterm(ser, term):
        l=ser.s
        l.append(term)
        return series(l)
        
    def eval(ser, time):
        l=ser.s
        val=arb('0')
        for i in range(len(l)):
            val+=l[i] * time**i
        return val
        
    def convrad(serlist, deg):
        final10=[]
        for i in range(10):
            cinflist = [float(ser.s[deg-i]) for ser in serlist]
            infnorm = np.max(np.abs(cinflist))
            final10.append(infnorm**(1/(deg-i)))
        rad=1/max(final10)
        
        return arb(rad)
                
               
def Mseries(frob, t0, prevseries):
    
    etas=[]
    for i in range(6):
        etai=series(prevseries[i].s)
        etas.append(etai)

    [eta1,eta2,eta3,eta4,eta5,eta6]=etas
    t=series([arb('0'),arb('1')])
    M1= -eta1*eta4
    M2= b1**2/h**3 + b1/h**2*eta5 - b1/h*eta2 - eta2*eta5
    M3= b1**2/h**3 - b1/h**2*eta6 + b1/h*eta3 - eta3*eta6
    
    if frob==True:
        t2 = -b1/h**2+eta2
        t3 = b1/h**2+eta3
        
        if eta1.deg==1:
            X1t=arb('1/2')
            X2=1/h
            X3=1/h
            inv2233=series.inv(X2*X2*X3*X3)
            
            R1 = - 1/2*inv2233 * (X2+X3)*(X2+X3) * (-2*b1/h**2 + eta2 - eta3)*(-2*b1/h**2 + eta2 - eta3)*X1t*X1t
            K=-1/(2*h**2)*(eta2+eta3)
            rest5 = X2*X2 + 1/2*(2*b1/h**2 + eta3 - eta2) * eta1 * (X3*X3+X2*X2)*(X3+X2)*inv2233 + 1/2*(2*b1/h**2 + eta3 - eta2) * inv2233 * K + arb('3')
            rest6 = X3*X3 - 1/2*(2*b1/h**2 + eta3 - eta2) * eta1 * (X3*X3+X2*X2)*(X3+X2)*inv2233 - 1/2*(2*b1/h**2 + eta3 - eta2) * inv2233 * K + arb('3')
        else:
            X1t= arb('1/2') + t*eta1
            X2 = 1/h - b1/h**2 * t + t*eta2
            X3 = 1/h + b1/h**2 * t + t*eta3
            
            X1tsq=X1t*X1t
            inv1t2=series.inv(X1tsq)
            X22=X2*X2
            X33=X3*X3
            X2233=X22*X33
            inv2233=series.inv(X2233)
            t22=t2*t2
            t33=t3*t3
            
            R1 = 1/2*t*t*X2233*inv1t2 - 1/2*inv2233 * (X2+X3)*(X2+X3) * (-2*b1/h**2 + eta2 - eta3)*(-2*b1/h**2 + eta2 - eta3)*X1tsq
            K=-1/(2*h**2)*(eta2+eta3) + 1/(2*h) *t*(eta2+eta3)*(eta2+eta3)+ arb('1/4') *t*(t22+t33)*(-2/h + t*(eta2+eta3)) - h*t*(2/h*t2 + t*t22)*(2/h*t3 + t*(t33))
            Kpart=1/2*(2*b1/h**2 + eta3 - eta2) * eta1 * (1+ t*eta1) * (X33+X22)*(X3+X2)*inv2233 + 1/2*(2*b1/h**2 + eta3 - eta2) * inv2233 * K 
            frontpart=-1/2 * t*t*X2233*inv1t2 + arb('3')
            rest5= frontpart + X22 + Kpart
            rest6= frontpart + X33 - Kpart
    
    if frob==False:
        X1 = series.inv(series.polyseries(1, arb('2'), eta1.deg) + 2*t0) + eta1
        X2 = 1/h - b1*t0/h**2 + series.polyseries(1, arb('1'), eta2.deg)*(-b1/h**2 + eta2) + t0*eta2
        X3 = 1/h + b1*t0/h**2 + series.polyseries(1, arb('1'), eta3.deg)*(b1/h**2 + eta3) + t0*eta3
        
        X11=X1*X1
        X22=X2*X2
        X33=X3*X3
        X231=X22*X33*series.inv(X11)
        X312=X33*X11*series.inv(X22)
        X123=X11*X22*series.inv(X33)
        
        R1 = 1/2*(X231 - X312 - X123) + X11
        
        R2 = 1/2*(X312 - X123 - X231) + X22
        
        R3 = 1/2*(X123 - X231 - X312) + X33
        
        back=series.inv(series.polyseries(1,arb('1'),eta2.deg)+t0)*(b1/h - arb('1/2')*h * (eta2-eta3))
        rest5 = R2 - back + arb('3')
        rest6 = R3 + back + arb('3')
        
    sum456=eta4+eta5+eta6
    M4 = -eta4*sum456 + arb('3') + R1
    M5 = -(b1/h + eta5)*sum456 + rest5
    M6= -(-b1/h + eta6)*sum456 + rest6
    
    
    return (M1,M2,M3,M4,M5,M6)
    
L=arb_mat([[-1,0,0,-1/2,0,0],[0,-1,0,0,-1/h,0],[0,0,-1,0,0,-1/h],[0,0,0,-2,-1,-1],[0,-h/2,h/2,0,-1,0],[0,h/2,-h/2,0,0,-1]])
    
def nextterm_frobenius(prevseries, currentdeg):
    b = [etai.s[currentdeg] for etai in Mseries(True, 0, prevseries)]
    nextterm = (arb_mat(((currentdeg+1)*np.eye(6)).tolist()) - L).solve(arb_mat([[j] for j in b]))
    return nextterm


def frobenius(order):
    deg=0
    prevseries=[series([0]),series([0]),series([0]),series([0]),series([0]),series([0])]
    while deg<order:
        prevcopy=[S for S in prevseries]
        nextone=nextterm_frobenius(prevseries, deg).tolist()
        newlist=[]
        for i in range(6):
            newlist.append(series.addnewterm(prevcopy[i], nextone[i][0]))
        prevseries=newlist
        deg+=1
    return prevseries, series.convrad(prevseries, deg)

def nextterm(prevseries, currentdeg, t0):
    b = arb_mat([[etai.s[currentdeg]] for etai in Mseries(False, t0, prevseries)])
    
    prevvectors=[[eta.s[deg] for eta in prevseries] for deg in range(currentdeg+1)]
    
    i=currentdeg
    gammai=arb('0')
    for l in range(i+1):
        gammai += (-1/t0)**arb(str(l)) * L * arb_mat([[x] for x in prevvectors[i-l]])
    nextterm = (1/t0*gammai + b)/arb(str(i+1))
        
    
    return nextterm

def nonfrobenius(order, initdata, t0):
    deg = 0
    prevseries=[series([val]) for val in initdata]
    while deg<order:
        nextone = nextterm(prevseries, deg, t0).tolist()
        newlist=[]
        for i in range(6):
            newlist.append(series.addnewterm(prevseries[i], nextone[i][0]))
        prevseries=newlist
        deg+=1
    return prevseries, series.convrad(prevseries, deg)

def solver(finaltime, order):
    ser, rad = frobenius(order)
    print('centre at time ', round(float(0), 6), ', radius of convergence ', round(float(rad), 6))
    t=rad/2
    allseries=[(arb('0'), ser)]
    prevtime=arb('0')
    while t<finaltime:
        etastart=[series.eval(etai, t-prevtime).mid() for etai in ser]
        ser, rad= nonfrobenius(order, etastart, t)
        allseries.append((t,ser))
        print('centre at time ', round(float(t), 6), ', radius of convergence ', round(float(rad), 6))
        prevtime=t
        t+=rad/2
    print('solver done at time '+str(round(time.time()-startclock,3)) + 's')
    return allseries
    
    
def eqrhs(eta, t, verbose=False):
    [eta1,eta2,eta3,eta4,eta5,eta6]=eta
    if t==0:
        raise Exception('can\'t evaluate the RHS of the equation at t=0!')
    else:
        singterm = (1/t * L * arb_mat([[v] for v in eta])).tolist()

        M1= -eta1*eta4
        M2= b1**2/h**3 + b1/h**2*eta5 - b1/h*eta2 - eta2*eta5
        M3= b1**2/h**3 - b1/h**2*eta6 + b1/h*eta3 - eta3*eta6
    
        X1=1/(2*t) + eta1
        X2=1/h - b1*t/h**2 + t*eta2
        X3=1/h + b1*t/h**2 + t*eta3
    
        R1=1/2*(X2**2*X3**2/X1**2 - X3**2*X1**2/X2**2 - X1**2*X2**2/X3**2)+X1**2
        R2=1/2*(X3**2*X1**2/X2**2 - X1**2*X2**2/X3**2 - X2**2*X3**2/X1**2)+X2**2
        R3=1/2*(X1**2*X2**2/X3**2 - X2**2*X3**2/X1**2 - X3**2*X1**2/X2**2)+X3**2
    
        M4 = -eta4*(eta4+eta5+eta6) + R1 + arb('3')
        M5 = -(b1/h + eta5)*(eta4+eta5+eta6) + R2 - 1/t*(b1/h - 1/2*h*(eta2-eta3)) + arb('3')
        M6= -(-b1/h + eta6)*(eta4+eta5+eta6) + R3 - 1/t*(-b1/h + 1/2*h*(eta2-eta3)) + arb('3')
    return [singterm[0][0]+M1, singterm[1][0]+M2, singterm[2][0]+M3, singterm[3][0]+M4, singterm[4][0]+M5, singterm[5][0]+M6]

class eta(object):
    def __init__(self, order, tf):
        self.tf=tf
        self.deg=order
        self.sols=solver(tf, order)
        self.timeslist = [tu[0] for tu in self.sols]
    
    def __call__(self, t):
        tindex = np.searchsorted(self.timeslist, t)-1
        tstart=self.timeslist[tindex]
        segment=self.sols[tindex][1]
        vals=[]
        for i in range(6):
            val=series.eval(segment[i], t-tstart)
            vals.append(val)
        return vals
        
    def deriv(self, t):
        return eqrhs(self(t), t)
        
    def chebfit(self, ordercheb):
        etaps=()
        for i in range(6):
            etap = chebyfit(lambda t: eqrhs(self(t), t)[i].mid(), arb('0'), self.tf, ordercheb)
            etaps+=(etap,)
        self.eta_prime=etaps
        etas=()
        for i in range(6):
            eta = self.eta_prime[i].integral()
            etas+=(eta,)
        self.eta_cheb=etas
        return self.eta_prime, self.eta_cheb
    
    def p(self, t):
        result=np.array([arb('0') for i in range(6)])
        for i in range(6):
            result[i]=self.eta_cheb[i](t)
        return result
    
    def p_deriv(self, t):
        result=np.array([arb('0') for i in range(n)])
        for i in range(6):
            result[i]=self.eta_prime[i](t)
        return result
    
    def difference(self, t):
        rhs = eqrhs(self.p(t), t)
        lhs = self.p_deriv(t)
        return [rhs[i]-lhs[i] for i in range(6)]
        
    def chebRHS(self):
        tf=arb(str(self.tf))
        t=arb_chebyT(arb_poly([arb('1/2')*tf, arb('1/2')*tf]), a=arb('0'), b=tf)
        eta1, eta2, eta3, eta4, eta5, eta6 = self.eta_cheb
        
        M1= -eta1*eta4
        M2= b1**2/h**3 + b1/h**2*eta5 - b1/h*eta2 - eta2*eta5
        M3= b1**2/h**3 + (-b1/h**2*eta6) + b1/h*eta3 - eta3*eta6
        
        X1t= arb('1/2') + t*eta1
        X2 = 1/h +(- b1/h**2 * t) + t*eta2
        X3 = 1/h + b1/h**2 * t + t*eta3
        
        t2 = -b1/h**2+eta2
        t3 = b1/h**2+eta3
        
        num1=arb('1/2')*X2*X2*X3*X3*t*t
        denom1=X1t*X1t
        num2= -arb('1/2')*(X2+X3)*(X2+X3)*(-2*b1/h**2 + eta2 - eta3)*(-2*b1/h**2 + eta2 - eta3)*X1t*X1t
        denom2=X2*X2*X3*X3
        R1 = T_rational(num1, denom1) + T_rational(num2, denom2)
        
        K=-1/(2*h**2)*(eta2+eta3) + 1/(2*h) *t*(eta2+eta3)*(eta2+eta3)+ arb('1/4') *t*(t2*t2+t3*t3)*(-2/h + t*(eta2+eta3)) +(- h)*t*(2/h*t2 + t*t2*t2)*(2/h*t3 + t*(t3*t3))
        
        
        Kpart=T_rational(1/2*(2*b1/h**2 + eta3 + (-eta2)) * eta1 * (1+ t*eta1) * (X3*X3+X2*X2)*(X3+X2) + 1/2*(2*b1/h**2 + eta3 + (-eta2)) * K, denom2)
        frontpart=T_rational(arb('-1/2') * t*t*denom2, X1t*X1t) + arb('3')
        rest5= frontpart + X2*X2 + Kpart
        rest6= frontpart + X3*X3 +(- Kpart)
        sum456=eta4+eta5+eta6
        
            
        M4 = -eta4*sum456 + arb('3') + R1
        M5 = -(b1/h + eta5)*sum456 + rest5
        M6= -(-b1/h + eta6)*sum456 + rest6
        self.MRHS = (M1, M2, M3, M4, M5, M6)
        firstterm=()
        for i in range(6):
            term=arb('0')
            for j in range(6):
                term+=L.table()[i][j]*self.eta_cheb[j]

            firstterm+=(T_rational(term, t),)
        self.chebRHS = tuple([self.MRHS[i] + firstterm[i] for i in range(6)])
        
        self.eta_prime_mon = [self.eta_prime[i].to_mono() for i in range(6)]
        self.eta_cheb_mon = [self.eta_cheb[i].to_mono() for i in range(6)]
        self.chebRHS_num_mon = [self.chebRHS[i].num.to_mono() for i in range(6)]
        self.chebRHS_denom_mon = [self.chebRHS[i].denom.to_mono() for i in range(6)]       
        
    def cheberror(self):
        diffscheb = [self.eta_prime[i] - self.chebRHS[i] for i in range(6)]
        diffsmon = [T_rational(expr.num.to_mono(), expr.denom.to_mono()) for expr in diffscheb]
        C0gaps = [self.C0gen(expr, 0.01) for expr in diffsmon]
        C0eps = arb.sqrt(C0gaps[0]**2 + C0gaps[1]**2 + C0gaps[2]**2 + C0gaps[3]**2 + C0gaps[4]**2 + C0gaps[5]**2)
        L2eps = C0eps * self.tf
        self.C0error = C0eps
        return C0eps
    
    def polybool(self, poly, eps): #gives True if poly is uniformly bounded by eps, and False otherwise
        tf = arb(str(self.tf))
        pluspoly = poly + eps
        minuspoly = poly - eps
        diffp = sturmvars(pluspoly, arb('0'), tf, 0.5)
        diffm = sturmvars(minuspoly, arb('0'), tf, 0.5)
        tmid = arb('1/2')*tf
        midbool = poly(tmid).abs_upper()<eps
        if diffp==0 and diffm==0 and midbool==True:
            return True
        else:
            return False
    def rationalbool(self, rat, eps):
        tf = arb(str(self.tf))
        denom = rat.denom
        num = rat.num
        diffp = sturmvars(num + denom*eps, arb('0'), tf, 0.5)
        diffm = sturmvars(num - denom*eps, arb('0'), tf, 0.5)
        tmid = arb('1/2')*tf
        midbool = rat(tmid).abs_upper()<eps
        if diffp==0 and diffm==0 and midbool==True:
            return True
        else:
            return False
        
        
    def C0gen(self, expr, thresh): #gives a reasonable C^0 upper bound on a polynomial or a rational function using binary search and Sturm's theorem.
        tf = arb(str(self.tf))
        lower = arb('-10')
        higher = arb('2')
        try:            
            while self.polybool(expr, arb('10')**lower) is True:
                lower-=100
            while self.polybool(expr, arb('10')**higher) is False:
                higher+=100 
            while higher-lower>arb(str(thresh)):
                testexp = arb('1/2')*(lower+higher)
                testbool = self.polybool(expr, arb('10')**testexp)
                if testbool is True:
                    higher = testexp
                else:
                    lower = testexp
            return arb('10')**higher
        except:
            while self.rationalbool(expr, arb('10')**lower) is True:
                lower-=100
            while self.rationalbool(expr, arb('10')**higher) is False:
                higher+=100 
            while higher-lower>arb(str(thresh)):
                testexp = arb('1/2')*(lower+higher)
                testbool = self.rationalbool(expr, arb('10')**testexp)
                if testbool is True:
                    higher = testexp
                else:
                    lower = testexp
            return arb('10')**higher
        
    
    def Mlbound(self):
        thr=0.01
        e4 = self.C0gen(self.eta_prime_mon[3], thr)
        b1p5 = self.C0gen(b1/h + self.eta_prime_mon[4], thr)
        b1m6 = self.C0gen(b1/h - self.eta_prime_mon[5], thr)
        M00 = arb.max(arb.max(e4, b1p5), b1m6)
        M01 = arb.max(arb.max(self.C0gen(self.eta_prime_mon[0], thr), self.C0gen(b1/h**2 - self.eta_prime_mon[1], 0.1)), self.C0gen(b1/h**2 + self.eta_prime_mon[2], thr))
        sum456norm = self.C0gen(self.eta_prime_mon[3]+self.eta_prime_mon[4]+self.eta_prime_mon[5], thr)
        
        M11 = sum456norm + arb('3').sqrt()*arb.sqrt(e4**2 + b1p5**2 + b1m6**2)
        
        #Now we write out all the terms of M10. We will call them mij where i=4,5,6, and j=1,2,3
        t=arb_poly([arb('0'), arb('1')])
        t3 = arb_poly([arb('0'), arb('0'), arb('0'), arb('1')])
        eta1 = self.eta_cheb_mon[0]
        eta2 = self.eta_cheb_mon[1]
        eta3 = self.eta_cheb_mon[2]
        X2 = 1/h + (- b1/h**2 * t ) + t*self.eta_cheb_mon[1]
        X3 = 1/h + b1/h**2 * t + t*self.eta_cheb_mon[2]
        X2sq=X2*X2
        X3sq=X3*X3
        X23sq = X2sq*X3sq
        half1=arb('1/2') + t*eta1
        half1sq = half1*half1
        diffterm = 2*b1/h**2 + eta3 + (-eta2)
        
        bn2sq= (-b1/h**2 + eta2)*(-b1/h**2 + eta2)
        bn2cu= bn2sq*(-b1/h**2 + eta2)
        bn3sq= (b1/h**2 + eta3)*(b1/h**2 + eta3)
        bn3cu= bn3sq*(b1/h**2 + eta3)
        
        m41 = self.C0gen( -T_rational(-X2*X3 * t3, half1sq*half1), thr) + self.C0gen( - T_rational(t*half1*diffterm*(X2+X3)*(X2+X3), X23sq), thr)
        
        m42 = self.C0gen( T_rational(X2*X3sq * t3, half1sq), thr) + self.C0gen( T_rational( diffterm*(X2sq + X3sq)*(X2 + X3)*half1sq, X23sq*X2), thr)
        
        m43 = self.C0gen( T_rational(X2sq*X3 * t3, half1sq), thr) + self.C0gen( T_rational( -diffterm*(X2sq + X3sq)*(X2 + X3)*half1sq, X23sq*X2), thr)
        
        m51 = self.C0gen( T_rational(X23sq * t3, half1sq*half1) + T_rational(half1 * (X2sq+X3sq)*(X3+X2)*diffterm, X23sq), thr)
        
        back52_62 = self.C0gen(T_rational( (-5*b1/h**3 + 1/h * (3*eta2 + (-2*eta3))) + t*(3*bn2sq - bn3sq) + t*t*h* bn2cu, 4*X2sq*X2) +T_rational(3*b1/h**2 - eta2 + eta3 + h*t*bn3sq, 4*X3sq), thr)
        
        m52 = self.C0gen( 2*X2*t - T_rational(X3sq*X2*t3, half1sq) - eta1*(1+t*eta1)*(T_rational(X3sq, X2*X2*X2) + T_rational(X2, X3sq)), thr) + back52_62 
        
        back53_63 = self.C0gen(T_rational( (5*b1/h**3 + 1/h * (3*eta3 + (-2*eta2))) + t*(3*bn3sq - bn2sq) + t*t*h* bn3cu, -4*X3sq*X3) + T_rational(-3*b1/h**2 - eta3 + eta2 + h*t*bn2sq, -4*X2sq), thr)
        m53 = self.C0gen(-T_rational(X2sq*X3 * t*t*t, half1sq) + eta1*(1+t*eta1)*(T_rational(X2sq, X3*X3*X3) + T_rational(X3, X2sq)), thr) + back53_63
        
        m61 = self.C0gen( T_rational(X23sq*t*t*t, half1sq*half1) - T_rational(half1 * (X2sq+X3sq)*(X2+X3)*diffterm, X23sq), thr)
        
        m62 = self.C0gen( T_rational(-X2*X3sq * t*t*t, half1sq) +  eta1*(1+t*eta1)*(T_rational(X3sq, X2*X2*X2) + T_rational(X2, X3sq)), thr) + back52_62
        
        m63 = self.C0gen(2*X3*t - T_rational(X23sq * t*t*t, half1sq*half1) - eta1*(1+t*eta1)*(T_rational(X2sq, X3*X3*X3) + T_rational(X3, X2sq)), thr) + back53_63
        
        M10 = arb.sqrt(m41**2 + m42**2 + m43**2 + m51**2 + m52**2 + m53**2 + m61**2 + m62**2 + m63**2)
        
        self.Cl = M00 + M01 + M10 + M11
        return M00 + M01 + M10 + M11

    def Mnlbound(self):
        thr = 0.01
        t=arb_poly([arb('0'), arb('1')])
        eta1 = self.eta_cheb_mon[0]
        eta2 = self.eta_cheb_mon[1]
        eta3 = self.eta_cheb_mon[2]
        X2 = 1/h + (- b1/h**2 * t ) + t*self.eta_cheb_mon[1]
        X3 = 1/h + b1/h**2 * t + t*self.eta_cheb_mon[2]
        X2sq=X2*X2
        X3sq=X3*X3
        X23sq = X2sq*X3sq
        half1=arb('1/2') + t*eta1
        half1sq = half1*half1
        
        A1 = arb('3/2') + arb('3/2')*self.C0gen(T_rational(2*t*t*t*X2*X3*(X2+X3), half1sq) + T_rational(X3sq*X2 + 2*half1* (X2*X3 + X3sq), X2sq*X2), thr) + self.C0gen( T_rational(X2sq*X3 + 2*half1* (X2*X3 + X2sq), X3sq*X3), thr)
        
        A2 = arb('1/2') + arb('3/2')*(self.C0gen(arb('2/3')*t**2 + T_rational(t*t*t*t*X3sq + 2*t*t*t*X2*X3sq, half1sq)  +2*X2*X3*t*t + T_rational(2*X2*half1*(half1+arb('1')), X3sq), thr) + self.C0gen( T_rational(half1sq*(3*X3sq + 2*X3*X2sq) + 2*half1 * X3sq*X2, X2sq*X2sq) , thr))
        
        A3 = arb('1/2') + arb('3/2')*(self.C0gen(arb('2/3')*t**2 + T_rational(t*t*t*t*X2sq + 2*t*t*t*X3*X2sq, half1sq)  +2*X2*X3*t*t + T_rational(2*X3*half1*(half1+arb('1')), X2sq), thr) + self.C0gen(T_rational(half1sq*(3*X2sq + 2*X2*X3sq) + 2*half1 * X2sq*X3, X3sq*X3sq) , thr))
        
        self.Cnl = arb.max(arb.max(A1, arb('7/2')), arb.max(A2,A3))
        
        return arb.max(arb.max(A1, arb('7/2')), arb.max(A2,A3))

lamb = arb('1/2')/h * (h-arb('1'))**2

tfinal = arb('2.25')

print('started solver')

solu = eta(110, tfinal)
solu.chebfit(110)

solu.chebRHS()

print('C_nl is', float(solu.Mnlbound()))
print('C_l is', float(solu.Mlbound()))

print('the approximation error epsilon is ', float(solu.cheberror()))

B = arb('19')/arb('8') + arb('1')/arb.exp(arb('1')) + arb('1/2')*(h + arb('1')/h)

Cl = solu.Cl
Cnl = solu.Cnl

t0 = (arb.sqrt(arb('3'))/arb('2') * arb.exp(-(Cl + arb('1/2'))*tfinal - lamb)/(tfinal**arb.sqrt(lamb)*B*Cl))**(arb('1')/(arb('1')-lamb))

Itf = arb.exp((arb('2')*Cl+arb('1'))*tfinal + arb('2')*lamb)*(tfinal/t0)**(arb('2')*lamb)

epsilon = arb('1')/(arb('4')*Itf*Cnl*(arb.sqrt(arb('2')*tfinal) + arb('2')/arb.sqrt(arb('3'))*t0*B)**2)

print('epsilon0 is ', float(epsilon))
print('B is ', float(B))
print('t0 is ', float(t0))
print('I(tf) is ', float(Itf))

murad = arb('1')/(2*arb.sqrt(Itf)*Cnl*(arb.sqrt(arb('2')*tfinal) + (arb('2')/arb.sqrt(arb('3')))*t0*B))
mudiff = arb(mid = 0, rad = murad.abs_upper())
etaf = [solu.eta_cheb[i](tfinal) + mudiff for i in range(6)]
adotoa=solu.eta_cheb[3](tfinal)+1/tfinal
r_end = arb.tanh(tfinal/2)
rho_end = 1/2*(1-r_end**2)

print('LHS - RHS in hypothesis (3) of Lemma 3.2 is ', inf_try_from_sol((h, b1), etaf, tfinal, (arb('.375'), arb('.43'), arb('-2.5'))))

plt.show()
