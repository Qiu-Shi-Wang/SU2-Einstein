from flint import arb, arb_poly, ctx, acb_poly, acb, fmpz, fmpq
from tqdm import tqdm


class arb_chebyT(arb_poly):
    """Polynomials over the Chebyshev polynomial T_n basis rescaled
    to operate in the interval [a, b]."""

    def __init__(self, val=None, a=arb('-1'), b=arb('1')):
        super().__init__(val)
        self.a = arb(a)
        self.b = arb(b)
        self.monomials = None
        self.cesamials = None
        self.sub_poly = acb_poly(self.coeffs())

    def __call__(self, t):
        if isinstance(t, arb):
            return self.evaluate([t])[0]
        if isinstance(t, acb):
            return self.evaluate([t])[0]
        if isinstance(t, (int, float, fmpz, fmpq)):
            return self(arb(t))
        if isinstance(t, (complex)):
            return self(acb(t))
        raise TypeError("cannot call arb_chebyT with input of type %s", self.__class__.__name__, type(t))


    def derivative(self):
        # dT_n / dx = n U_{n-1}
        # U_n = 2 sum_{odd j}^n T_j if n is odd
        # U_n = 2 sum_{even j}^n T_j if n is even
        # Check if scalar
        scale = arb('2')/(self.b - self.a)
        if self.degree() == 0:
            return arb_poly([arb('0')])
        if self.degree() == 1:
            return arb_chebyT([self.coeffs()[1]], a=self.a, b=self.b) * scale
        U_coeffs = [(i+1)*x for (i, x) in enumerate(self.coeffs()[1:])]
        new_coeffs = [arb('0') for i in range(self.degree())]
        for idx, c in enumerate(U_coeffs):
            new_coeffs[idx] += 2*sum(U_coeffs[idx::2])
        new_coeffs[0] -= sum(U_coeffs[0::2])
        return arb_chebyT(new_coeffs, a=self.a, b=self.b) * scale
    
    def evaluate(self, xs, algorithm='fast'):
        scale = arb('2')/(self.b-self.a)
        end_pt = arb('-1')*(self.b+self.a)/(self.b-self.a)
        xs_c = [acb(scale*x +end_pt) for x in xs]
        x1 = [x - (x*x - 1).sqrt() for x in xs_c]
        x2 = [x + (x*x - 1).sqrt() for x in xs_c]
        y1 = self.sub_poly.evaluate(x1, algorithm=algorithm)
        y2 = self.sub_poly.evaluate(x2, algorithm=algorithm)
        out = []
        for idx, x in enumerate(xs):
            u = (y1[idx] + y2[idx])/arb('2')
            if isinstance(x, acb) or isinstance(x, complex):
                out.append(u)
            else:
                out.append(u.real)
        return out
    
    def integral(self):
        new_coeffs = [arb('0') for idx in range(self.degree() + 2)]
        for idx, c in enumerate(self.coeffs()):
            aidx = arb(str(idx))
            if idx == 0:
                # integral of T_0 is T_1
                new_coeffs[1] += c
            elif idx == 1:
                # integral of T_1 is x**2 / 2 = 1/4*T_2 + 1/4*T_0
                new_coeffs[2] += c / arb('4')
                new_coeffs[0] += c / arb('4')
            else:
                # integral of T_n = 1/2(T_{n+1}/(n+1) - T_{n-1}/(n-1))
                #                 + n/(n^2 - 1) * sin(pi n / 2)
                # ignore constant coeff, will be removed later
                new_coeffs[idx+1] += (c / arb('2')) / (aidx + 1)
                new_coeffs[idx-1] -= (c / arb('2')) / (aidx - 1)
        scale = (self.b - self.a) * arb('0.5')
        new_poly = arb_chebyT(new_coeffs, a=self.a, b=self.b)
        # Integrating from 0 to x, so should be zero at 0
        return (new_poly - new_poly(arb('0'))) * scale
    
    def _compute_cesamials(self):
        """Compute the Cesamials defined by R_n(x) = (T_n(ax + b) - T_n(b)) / x"""
        if self.cesamials is not None:
            return
        N = self.degree() + 1
        scale = arb('2')/(self.b - self.a)
        end_pt = arb('-1')*(self.b + self.a)/(self.b - self.a)
        R = [arb_chebyT([arb('0')], a=self.a, b=self.b), 
             arb_chebyT([scale], a=self.a, b=self.b)]
        # Recurrence: R[n](x) = 2*a*T[n](x)+2*b*R[n-1](x)-R[n-2](x)
        for k in range(N):
            R_tmp = (2 * end_pt * R[-1] - R[-2]).coeffs()
            if len(R_tmp) == 0:
                R_tmp = [arb('0')]
            R.append(arb_chebyT(R_tmp + [2*scale], a=self.a, b=self.b))
        self.cesamials = R
        
    def cesaro(self):
        """Compute the Cesaro mean $\\frac{1}{x}\\int_{0}^x T_n(x) dx$ in
        terms of Chebyshev polynomials."""
        self._compute_cesamials()
        scale = arb('2')/(self.b-self.a)
        end_pt = arb('-1')*(self.b+self.a)/(self.b-self.a)
        poly = arb_chebyT([arb('0')],a=self.a,b=self.b)
        for idx, c in enumerate(self.coeffs()):
            aidx = arb(str(idx))
            if idx == 0:
                # integral of T_0 / x is T_0
                poly += arb_chebyT([arb('1')],a=self.a,b=self.b) * c
            elif idx == 1:
                # integral of T_1 / x is 1/2*T_1 + b/2*T_0
                poly += arb_chebyT([arb('0.5')*end_pt, arb('0.5')],a=self.a,b=self.b) * c
            else:
                # integral of T_n / x is 1/a[1/(2(n+1))R_{n+1}(x) - 1/2(n+1)R_{n-1}(x)]
                tmp_poly = self.cesamials[idx+1] * (arb('1')/(2*(aidx+1)))
                tmp_poly -= self.cesamials[idx-1]* (arb('1')/(2*(aidx-1)))
                tmp_poly *= arb('1')/scale
                poly += tmp_poly * c
        return poly

    def _compute_cheb_mono(self):
        if self.monomials is not None:
            return
        N = self.degree() + 1
        scale = arb('2')/(self.b - self.a)
        end_pt = arb('-1')*(self.b + self.a)/(self.b - self.a)
        T = [arb_poly([arb('1')]), arb_poly([end_pt, scale])]
        # Recurrence: T[n+1](ax+b) = 2*(ax+b)*T[n](ax+b) - T[n-1](ax+b)
        step_poly = arb_poly([2*end_pt, 2*scale])
        for k in range(N-2):
            T.append(step_poly * T[-1] - T[-2])
        self.monomials = T

    def to_mono(self):
        """Return the representation of the polynomial in terms of the
        monomial basis."""
        # From mpmath implementation
        self._compute_cheb_mono()
        c = self.coeffs()
        d = arb_poly([arb('0')])
        for (ck, Tk) in zip(c, self.monomials):
            d += ck * Tk
        return d
    
    def __str__(self):
        out_str = ''
        for idx, c in enumerate(self.coeffs()[::-1]):
            if c == 0:
                continue
            out_str += '({coeff})*T_{idx}(x) + '.format(coeff=c, idx=self.degree()-idx)
        return out_str[:-3]
    
    def __repr__(self):
        return str(self)
    
    def __neg__(self):
        return arb_chebyT(super().__neg__().coeffs(), a=self.a, b=self.b)

    def __add__(self, other):
        if isinstance(other, arb_chebyT):
            if self.a != other.a or self.b != other.b:
                raise NotImplementedError('Cannot add polynomials over different supports')
        return arb_chebyT(super().__add__(other).coeffs(), a=self.a, b=self.b)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, arb_chebyT):
            if self.a != other.a or self.b != other.b:
                raise NotImplementedError('Cannot subtract polynomials over different supports')
        return arb_chebyT(super().__sub__(other).coeffs(), a=self.a, b=self.b)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        # T_m T_n = 1/2 (T_{m+n} + T_{|m-n|})
        poly1 = arb_poly(self.coeffs())
        if isinstance(other, arb_poly):
            if self.a != other.a or self.b != other.b:
                raise NotImplementedError('Cannot multiply polynomials over different supports')
            poly2 = arb_poly(other.coeffs())
            term1 = poly1*poly2 * arb('0.5')
            new_coeffs = term1.coeffs()
            for idx, c1 in enumerate(self.coeffs()):
                for idy, c2 in enumerate(other.coeffs()):
                    new_coeffs[abs(idx-idy)] += c1*c2 * arb('0.5')
            return arb_chebyT(new_coeffs, a=self.a, b=self.b)
        return arb_chebyT((poly1*other).coeffs(), a=self.a, b=self.b)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        raise NotImplementedError('Cannot perform powers of Chebyshev polynomials')
    
    def __divmod__(self, other):
        raise NotImplementedError('Cannot perform divmod of Chebyshev polynomials')
    
def chebyroots(a, b, N):
    """Compute the roots of the N-th degree Chebyshev polynomial"""
    h = arb('0.5')
    aN = arb(str(N))
    gap = (b - a)*h
    mid = (b + a)*h
    unif = [(arb(str(k))-h)/aN for k in range(1, N+1)]
    grid = [x.cos_pi()*gap + mid for x in unif]
    return grid

def chebyfit(f, a, b, N, verbose=False):
    """Fit a Chebyshev polynomial to a given function f on [a,b] to 
    arbitrary precision using ball arithmetic. Inputs are given as
    mpmath values. The function takes inputs as mpmath objects.
    
    Coefficients are computed using:
    http://mathworld.wolfram.com/ChebyshevApproximationFormula.html
    """
    # Compute Chebyshev grid
    grid = chebyroots(a, b, N)
    h = arb('0.5')
    aN = arb(str(N))
    if verbose:
        pbar = tqdm(grid)
    else:
        pbar = grid
    f_grid = [f(x) for x in pbar]
    coeffs = []
    for j in range(N):
        s = sum([f_grid[k-1] * ((arb(str(j))*(k-h))/aN).cos_pi() for k in range(1,N+1)])
        coeffs.append(arb('2') * s / aN)
    coeffs[0] -= coeffs[0] / arb('2')
    return arb_chebyT(coeffs, a=a, b=b)
    
    
#From this point on new lines of code are modifications by Qiu Shi
class T_rational(arb_chebyT):
    def __init__(self, num, denom):
        self.num = num
        if denom==0:
            raise Exception('rational functions cannot have zero denominator!')
        self.denom = denom
    def __call__(self, t):
        return self.num(t)/self.denom(t)
    def __neg__(self):
        return T_rational(-self.num, self.denom)
    def __add__(self, other):
        if self.__class__!=T_rational:
            return T_rational.__add__(other, self)
        if other.__class__==T_rational:
            if self.denom==other.denom:
                return T_rational(self.num+other.num, self.denom)
            else:
                return T_rational(self.num*other.denom + other.num * self.denom, self.denom*other.denom)
        elif other.__class__!=T_rational:
            return T_rational(self.num + other*self.denom, self.denom)
    def __radd__(self, other):
        return T_rational.__add__(self, other)
    def __sub__(self, other):
        return T_rational.__add__(self, -other)
    def __rsub__(obj1, obj2):
        return T_rational.__sub__(obj2, obj1)
        
    def __mul__(self, other):
        if other.__class__==T_rational:
            return T_rational(self.num*other.num, self.denom*other.denom)
        elif other.__class__!=T_rational:
            return T_rational(self.num*other, self.denom)
    def __rmul__(obj1, obj2):
        return T_rational.__mul__(obj1, obj2)
    def inv(self):
        if self.__class__==T_rational:
            return T_rational(self.denom, self.num)
        else:
            return T_rational(arb_chebyT(arb('1')), self.num)
    def __str__(self):
        return str(self.num) + ' / ' + str(self.denom)
    
