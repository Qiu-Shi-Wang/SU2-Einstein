import numpy as np
import scipy.integrate as int
import scipy
from flint import arb, ctx, acb

ctx.dps = 1000

t0=arb('10')**-4
L=arb('-3')

def phifunc(tau, params):
    (A,B,C,D)=params
    return B**2*D - arb('15')*A**2*tau/(arb('2')+tau)
    
def thetafunc(tau, params):
    (A,B,C,D)=params
    return arb('2')*C+arb('1')/B**2 - arb('5')*tau/(arb('2')+tau)/(A**2)
    
def alphafunc(tau, normZ2, s0, params):
    (A,B,C,D) = params
    term2 = (B**2*D - arb('15')*A**2)*(tau + s0) + arb('30')*A**2*arb.log(((arb('2')+tau)/(arb('2')-s0)).real)
    return normZ2 + term2

def expfactor(s, params):
    (A,B,C,D) = params
    intresult = (arb('2')*C + arb('1')/B**2 - arb('5')/A**2)*(-s) + arb('10')/A**2 * arb.log((arb('2')/(arb('2') + s)).real)
    return arb.exp(intresult.real)
    
def integrand2(s, normZ2, s0, params):
    return alphafunc(s, normZ2, s0, params)*thetafunc(s, params)*expfactor(s, params)

def gronwall_sumzbound(output, A,B,C):
    (s0,alphas0,betas0,gammas0,z1,z2,z3)=output
    K=arb('3/8')*(alphas0**4/betas0**4 + betas0**4/alphas0**4)/gammas0**4 + arb('3/8')*(alphas0**4/gammas0**4 + gammas0**4/alphas0**4)/betas0**4 + arb('3/8')*(betas0**4/gammas0**4 + gammas0**4/betas0**4)/alphas0**4
    print('K0 is ', K)
    normZ2=z1**2+z2**2+z3**2
    
    int2 = lambda x, _: integrand2(x, normZ2, s0, (A,B,C,K))
    
    gronfinal=alphafunc(arb('0'), normZ2, s0, (A,B,C,K)) + acb.integral(int2, -s0, arb('0'))
    return arb.sqrt(gronfinal.real)*arb.sqrt(arb('3'))
    
def inf_try_from_sol(initialdata, data, tf, params):
    (h, b1) = initialdata
    (eta1, eta2, eta3, eta4, eta5, eta6)=data
    Aval, Bval, Cval = params
    if 2*Cval+1/Bval**2<0:
        raise Exception('2C + 1/B^2 <0!')
    
    r0 = arb.tanh(tf/arb('2'))
    s0 = arb('1') - r0
    rho0 = arb('1/2')*(arb('1')-r0**2)
    alphas0 = rho0/(1/(arb('2')*tf) + eta1)
    betas0 = rho0/(1/h - b1*tf/h**2 + tf*eta2)
    gammas0 = rho0/(1/h + b1*tf/h**2 + tf*eta3)
    #print(Aval, Bval, Cval, eta1, eta2, eta3, eta4, eta5, eta6, r0, s0, rho0, alphas0, betas0, gammas0)
    Z1s0 = arb('1') + arb('1')/rho0 * (r0 - (arb('1')/tf + eta4))
    Z2s0 = arb('1') + arb('1')/rho0 * (r0 - (b1/h + eta5))
    Z3s0 = arb('1') + arb('1')/rho0 * (r0 - (-b1/h + eta6))
    Znorm = arb.sqrt(Z1s0**2+Z2s0**2+Z3s0**2)
    print('Z(s0) is ', Znorm)
    thetam = arb('2')*Cval + 1/Bval**2 + arb('5')*s0/(arb('2')-s0)/Aval**2
    hypo3apriori= arb('4')+Cval - s0/(arb('2')-s0)
    
    testoutput=(s0, alphas0, betas0, gammas0, Z1s0, Z2s0, Z3s0)
    g_sumz=gronwall_sumzbound(testoutput, Aval, Bval, Cval)
    if g_sumz/arb.sqrt(arb('3'))<arb('1/3'):
        print('zeta_m is ', g_sumz/arb.sqrt(arb('3')))
    else:
        raise Exception('\\zeta_m >= 1/3!')
    return g_sumz-hypo3apriori
