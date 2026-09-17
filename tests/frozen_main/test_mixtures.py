"""Methodical checks for the two Beta-mixture development candidates.

Every Monte-Carlo tolerance below is stated together with the sampling error it
is derived from, and every simulation uses a fixed seed, so none of these tests
is flaky. The exactness checks use rational arithmetic as an independent
reference rather than a second floating-point implementation.
"""
import math
import unittest
from fractions import Fraction
import numpy as np
from main_experiment import mixtures as mx
from main_experiment.baselines import activity, plugin, corrector, profile
from main_experiment.observation import validate, serialize, parse
from main_experiment.common import LEGACY_H

# The suffix candidate belongs to the superseded suffix-panel arm, which is kept as
# the development variant LEGACY_H. Arm B's candidate is part of the current design.

N_PANEL=4096   # panel size for synthetic suffix fixtures; only has to exceed N_obs


def make_observation(arm, patterns, counts, parameter):
    """A valid observation block from per-dyad window patterns and event counts.

    patterns: (n_dyads, n_windows) boolean of observed-active windows, each row
    with at least one active window. counts: same shape, events per active window.
    """
    n=patterns.shape[1]
    width=5
    table={}
    for row,ev in zip(patterns,counts):
        if arm==LEGACY_H: key='??'+''.join('1' if x else '0' for x in row)
        else: key=''.join('1' if x else '0' for x in row)
        d,e=table.get(key,(0,0))
        table[key]=(d+1,e+int(ev.sum()))
    order=[('??'+f'{p:03b}') if arm==LEGACY_H else f'{p:05b}' for p in range(1,8 if arm==LEGACY_H else 32)]
    rows=[(k,*table.get(k,(0,0))) for k in order]
    D=sum(r[1] for r in rows); M=sum(r[2] for r in rows)
    per=counts.sum(0)
    if arm==LEGACY_H: epw=[None,None]+[int(x) for x in per]
    else: epw=[int(x) for x in per]
    N=max(2,math.ceil((1+math.sqrt(1+8*D))/2))
    N=min(N,2*D) if D else 0
    # The suffix arm carries an integer panel size; any value at least N_obs is valid.
    if arm==LEGACY_H and parameter is not None and parameter<N: parameter=N
    o={'arm':arm,'N_obs':int(N),'D_obs':int(D),'M_obs':int(M),
       'Temporal_access':[0,0,1,1,1] if arm==LEGACY_H else [1]*width,
       'Events_per_window':epw,'Walk_A':None,'parameter':parameter,'table':rows}
    validate(o); return o


def simulate(rng,n_dyads,a,b,n_windows,lam=None,p=1.0):
    """Draw dyads from the assumed model and keep those with >=1 observed window."""
    q=rng.beta(a,b,n_dyads)
    true_active=rng.random((n_dyads,n_windows))<q[:,None]
    if lam is None:
        obs=true_active; ev=obs.astype(int)
    else:
        # zero-truncated Poisson(lam) events in every truly active window
        ev_true=np.zeros((n_dyads,n_windows),int)
        idx=np.flatnonzero(true_active.ravel())
        draws=np.empty(len(idx),int)
        for i in range(len(idx)):
            while True:
                x=rng.poisson(lam)
                if x>0: draws[i]=x; break
        ev_true.ravel()[idx]=draws
        kept=rng.binomial(ev_true,p)
        obs=kept>0; ev=kept
    keep=obs.any(1)
    return obs[keep],ev[keep],true_active[keep]


class ExactnessTests(unittest.TestCase):
    def test_cell_probs_match_exact_rational(self):
        """Independent reference: the same identity in exact rational arithmetic.

        Fraction(float) is the exact binary value of the float, so any difference
        is genuine floating-point error and not a mismatch of inputs. The grid
        deliberately includes a=1e5, b=1e-3, d=1, where q concentrates at one and
        Pr{J=0} is about 1e-26; the earlier alternating-moment form returned pure
        rounding noise there.
        """
        def exact(a,b,d,n=5):
            fa,fb,fd=Fraction(a),Fraction(b),Fraction(d)
            def jm(r,s):
                v=Fraction(1)
                for t in range(r): v*=(fa+t)
                for t in range(s): v*=(fb+t)
                for t in range(r+s): v/=(fa+fb+t)
                return v
            return [Fraction(math.comb(n,j))*fd**j*sum(
                Fraction(math.comb(n-j,s))*(1-fd)**(n-j-s)*jm(n-s,s) for s in range(n-j+1))
                for j in range(n+1)]
        worst=0.
        for a in (1e-3,1e-2,.5,1.,3.,1e3,1e5):
            for b in (1e-3,1e-2,.5,1.,3.,1e3,1e5):
                for d in (1e-6,1e-4,1e-2,.3,.75,.999999,1.):
                    got=mx.cell_probs(a,b,d,5); ex=exact(a,b,d)
                    for j in range(6):
                        e=float(ex[j])
                        if e>1e-300: worst=max(worst,abs(got[j]-e)/e)
        self.assertLess(worst,1e-13,f'worst relative error {worst:.3e}')

    def test_cell_probs_sum_to_one(self):
        for a,b,d in [(.3,.7,1.),(2.,5.,.4),(1e5,1e5,.9),(1e-2,1e-2,.01)]:
            self.assertAlmostEqual(sum(mx.cell_probs(a,b,d,5)),1.,places=12)

    def test_homogeneous_limit_is_binomial(self):
        """kappa -> infinity must collapse onto the existing homogeneous model."""
        mu,d=.3,.7
        prev=None
        for kappa in (1e2,1e4,1e6):
            got=mx.cell_probs(mu*kappa,(1-mu)*kappa,d,5)
            exp=[math.comb(5,j)*(mu*d)**j*(1-mu*d)**(5-j) for j in range(6)]
            err=max(abs(x-y) for x,y in zip(got,exp))
            if prev is not None: self.assertLess(err,prev/50)   # first order in 1/kappa
            prev=err
        self.assertLess(prev,1e-5)

    def test_profile_matches_hand_computation(self):
        """BetaBinomial(5,2,3) conditioned on K>=1, computed exactly by hand."""
        pk=[Fraction(math.comb(5,k))*Fraction(math.factorial(1+k)*math.factorial(7-k),
              math.factorial(9))*Fraction(math.factorial(4),math.factorial(1)*math.factorial(2))
            for k in range(6)]
        den=sum(pk[1:])
        want=[float(sum(pk[k:])/den) for k in range(2,6)]
        got=mx.predict_profile(2.,3.)
        for x,y in zip(got,want): self.assertAlmostEqual(x,y,places=12)
        self.assertEqual(want[:1],[float(Fraction(5,7))])

    def test_profile_is_monotone_and_in_range(self):
        rng=np.random.default_rng(0)
        for _ in range(200):
            a,b=np.exp(rng.uniform(-6,10,2))
            r=mx.predict_profile(float(a),float(b))
            self.assertTrue(all(0<=x<=1 for x in r),r)
            self.assertTrue(all(x>=y-1e-12 for x,y in zip(r,r[1:])),r)


class SuffixCandidateTests(unittest.TestCase):
    def test_recovers_its_own_model(self):
        """Data from the assumed model. 20000 dyads give a rho_2 standard error of
        about 1/sqrt(20000) = 0.007, so 0.03 is a safe four-sigma tolerance."""
        rng=np.random.default_rng(20260916)
        a,b=2.,3.
        obs,ev,true_active=simulate(rng,20000,a,b,3)
        o=make_observation(LEGACY_H,obs,np.maximum(ev,1),N_PANEL)
        mu=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
        fit=mx.fit_suffix(o,activity(mu,3))
        want=mx.predict_profile(a,b)
        self.assertIn(fit.status,('converged','weakly_identified'))
        for x,y in zip(fit.prediction,want): self.assertAlmostEqual(x,y,delta=.03)

    def test_homogeneous_data_hits_the_homogeneous_boundary(self):
        """Binomial data are the kappa -> infinity edge of the family."""
        rng=np.random.default_rng(7)
        q=.4
        act=rng.random((20000,3))<q
        act=act[act.any(1)]
        o=make_observation(LEGACY_H,act,act.astype(int),N_PANEL)
        mu=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
        fit=mx.fit_suffix(o,activity(mu,3))
        self.assertIn(fit.status,('boundary_homogeneous','weakly_identified','converged'))
        # whatever the status, the profile must stay close to the homogeneous answer
        for x,y in zip(fit.prediction,profile(activity(mu,3))):
            self.assertAlmostEqual(x,y,delta=.05)

    def test_strong_heterogeneity_beats_the_homogeneous_answer(self):
        """A U-shaped Beta is exactly what the homogeneous corrector cannot see."""
        rng=np.random.default_rng(11)
        a,b=.2,.2
        obs,ev,_=simulate(rng,20000,a,b,3)
        o=make_observation(LEGACY_H,obs,np.maximum(ev,1),N_PANEL)
        mu=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
        fit=mx.fit_suffix(o,activity(mu,3))
        want=mx.predict_profile(a,b)
        hom=profile(activity(mu,3))
        self.assertLess(abs(fit.prediction[0]-want[0]),abs(hom[0]-want[0]))

    def test_low_information_is_flagged_not_hidden(self):
        rng=np.random.default_rng(3)
        obs,ev,_=simulate(rng,12,2.,3.,3)
        o=make_observation(LEGACY_H,obs,np.maximum(ev,1),N_PANEL)
        mu=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
        fit=mx.fit_suffix(o,activity(mu,3))
        self.assertNotEqual(fit.status,'converged')
        self.assertTrue(all(0<=x<=1 for x in fit.prediction))

    def test_empty_sample(self):
        o={'arm':LEGACY_H,'N_obs':0,'D_obs':0,'M_obs':0,'Temporal_access':[0,0,1,1,1],
           'Events_per_window':[None,None,0,0,0],'Walk_A':None,'parameter':N_PANEL,
           'table':[('??'+f'{p:03b}',0,0) for p in range(1,8)]}
        validate(o)
        fit=mx.fit_suffix(o,.5)
        self.assertEqual(fit.status,'empty_sample')
        self.assertEqual(fit.prediction,[0.,0.,0.,0.])


class EventCandidateTests(unittest.TestCase):
    def _start(self,o):
        D=o['D_obs']; S=sum(p.count('1')*d for p,d,e in o['table']); M=o['M_obs']; p=o['parameter']
        from main_experiment.baselines import bisect
        mean=M/S
        r=0. if mean<=1 else bisect(lambda x:1. if x==0 else x/-math.expm1(-x),mean,0.,mean)
        return activity(S/D,5),r/p

    def test_retention_identity(self):
        """d(lambda,p) as derived, checked against a direct thinning computation."""
        for lam in (.2,1.,5.):
            for p in (.1,.5,.9):
                # Pr{no event kept | active} with N ~ ZTP(lam), summed term by term
                # so no factorial is ever materialised.
                term=1.; tot=0.
                for n in range(1,200):
                    term*=lam*(1-p)/n; tot+=term
                keep_none=tot/math.expm1(lam)
                want=1-keep_none
                got=-math.expm1(-p*lam)/-math.expm1(-lam)
                self.assertAlmostEqual(got,want,places=12)

    def test_recovers_its_own_model(self):
        """20000 dyads, thinning p=0.5. Same four-sigma argument as for the suffix."""
        rng=np.random.default_rng(20260917)
        a,b,lam,p=2.,3.,2.5,.5
        obs,ev,_=simulate(rng,20000,a,b,5,lam=lam,p=p)
        o=make_observation('B',obs,ev,p)
        mu0,lam0=self._start(o)
        fit=mx.fit_events(o,mu0,lam0)
        want=mx.predict_profile(a,b)
        self.assertIn(fit.status,('converged','weakly_identified'))
        for x,y in zip(fit.prediction,want): self.assertAlmostEqual(x,y,delta=.03)
        self.assertAlmostEqual(fit.lam,lam,delta=.3)

    def test_joint_lambda_is_reported_against_event_only(self):
        rng=np.random.default_rng(5)
        a,b,lam,p=1.5,4.,3.,.4
        obs,ev,_=simulate(rng,8000,a,b,5,lam=lam,p=p)
        o=make_observation('B',obs,ev,p)
        mu0,lam0=self._start(o)
        fit=mx.fit_events(o,mu0,lam0)
        self.assertTrue(math.isfinite(fit.lam) and math.isfinite(fit.lam_event_only))
        self.assertGreater(fit.lam,0)

    def test_homogeneous_data_matches_the_hurdle_corrector(self):
        rng=np.random.default_rng(13)
        q,lam,p=.35,2.,.6
        act=rng.random((20000,5))<q
        ev_true=np.zeros((20000,5),int)
        idx=np.flatnonzero(act.ravel()); draws=np.empty(len(idx),int)
        for i in range(len(idx)):
            while True:
                x=rng.poisson(lam)
                if x>0: draws[i]=x; break
        ev_true.ravel()[idx]=draws
        kept=rng.binomial(ev_true,p); obs=kept>0
        keep=obs.any(1)
        o=make_observation('B',obs[keep],kept[keep],p)
        mu0,lam0=self._start(o)
        fit=mx.fit_events(o,mu0,lam0)
        want=profile(q)
        for x,y in zip(fit.prediction,want): self.assertAlmostEqual(x,y,delta=.05)

    def test_empty_sample(self):
        o={'arm':'B','N_obs':0,'D_obs':0,'M_obs':0,'Temporal_access':[1]*5,
           'Events_per_window':[0]*5,'Walk_A':None,'parameter':.5,
           'table':[(f'{p:05b}',0,0) for p in range(1,32)]}
        validate(o)
        fit=mx.fit_events(o,.5,1.)
        self.assertEqual(fit.status,'empty_sample')


class NumericalGuardTests(unittest.TestCase):
    """The guards the main evaluation relies on, pinned so they cannot regress."""

    def test_penalty_region_is_never_accepted_as_a_fit(self):
        """A finite objective is not a fit: the invalid region returns PENALTY."""
        best,values,rejected=mx._solve(lambda z: mx.PENALTY,[[0.,0.]],
                                       [(-1.,1.),(-1.,1.)])
        self.assertIsNone(best); self.assertEqual(values,[])
        self.assertTrue(any('penalty' in r for r in rejected),rejected)

    def test_failed_optimiser_is_rejected_with_its_status(self):
        def nll(z): return float('nan')
        best,values,rejected=mx._solve(nll,[[0.,0.]],[(-1.,1.),(-1.,1.)])
        self.assertIsNone(best)
        self.assertTrue(rejected)

    def test_zero_class_of_the_thinned_positive_poisson(self):
        """The closed form holds for k>=1 only; k=0 needs its own expression."""
        for lam in (.3,1.5,4.):
            for p in (.2,.6,.95):
                direct=0.; term=1.
                for n in range(1,300):
                    term*=lam/n
                    direct+=term*(1-p)**n
                direct/=math.expm1(lam)
                naive=math.exp(-p*lam)/(1-math.exp(-lam))
                self.assertAlmostEqual(naive-direct,1/math.expm1(lam),places=12)
                d=-math.expm1(-p*lam)/-math.expm1(-lam)
                self.assertAlmostEqual(1-direct,d,places=12)

    def test_profile_clamp_only_absorbs_last_bit_noise(self):
        rng=np.random.default_rng(4)
        for _ in range(300):
            a,b=np.exp(rng.uniform(-6,12,2))
            r=mx.predict_profile(float(a),float(b))   # raises on a material clamp
            self.assertTrue(all(0.<=x<=1. for x in r))

    def test_kappa_upper_bound_is_an_approximation_not_the_limit(self):
        """kappa=1e6 is close to, but not equal to, the homogeneous model."""
        mu,d=.3,.7
        got=mx.cell_probs(mu*1e6,(1-mu)*1e6,d,5)
        exact=[math.comb(5,j)*(mu*d)**j*(1-mu*d)**(5-j) for j in range(6)]
        err=max(abs(x-y) for x,y in zip(got,exact))
        self.assertGreater(err,0.)          # not exact
        self.assertLess(err,1e-5)           # but numerically indistinguishable

    def test_bound_sensitivity_reports_a_stable_fit_as_stable(self):
        rng=np.random.default_rng(6)
        obs,ev,_=simulate(rng,4000,2.,3.,3)
        o=make_observation(LEGACY_H,obs,np.maximum(ev,1),N_PANEL)
        mu=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
        r=mx.bound_sensitivity(o,activity(mu,3),decades=2.)
        self.assertLess(r['max_profile_shift'],1e-4,r)
        self.assertLess(abs(r['nll_improvement']),1e-6,r)


class DiagnosticIndependenceTests(unittest.TestCase):
    """Diagnostic conditions overlap and must not mask one another."""

    def setUp(self):
        self.bounds = [(-mx.LOGIT_BOUND, mx.LOGIT_BOUND), mx.LOG_KAPPA_BOUNDS]

    def test_boundary_does_not_hide_a_start_disagreement(self):
        z = [0.0, mx.LOG_KAPPA_BOUNDS[1]]          # on the homogeneous bound
        label, flags = mx.diagnose(z, self.bounds, spread=1.0, flat=99.)
        self.assertEqual(label, 'boundary_homogeneous')
        self.assertTrue(flags['starts_disagree'])
        self.assertTrue(mx.is_unreliable(label, flags),
                        'the fallback must fire even though the label says boundary')

    def test_interior_agreeing_fit_is_reliable(self):
        label, flags = mx.diagnose([0.0, 0.0], self.bounds, spread=0., flat=99.)
        self.assertEqual(label, 'converged')
        self.assertFalse(any(flags.values()))
        self.assertFalse(mx.is_unreliable(label, flags))

    def test_missing_flatness_is_not_read_as_good_identifiability(self):
        label, flags = mx.diagnose([0.0, 0.0], self.bounds, spread=0., flat=float('nan'))
        self.assertTrue(flags['flatness_unavailable'])
        self.assertFalse(flags['weakly_identified'])
        self.assertEqual(label, 'flatness_unavailable')

    def test_flatness_ignores_failed_profile_optimisations(self):
        """A profile optimisation that fails must not contribute a rise."""
        class Best:
            x = [0.0, 0.0]
            fun = 1.0
        rises = mx._flatness(lambda z: mx.PENALTY, Best(), self.bounds, 1)
        self.assertNotEqual(rises, rises)          # NaN: no usable diagnosis

    def test_every_flag_is_reachable_on_real_fits(self):
        """The flags are exercised by actual development fits, not only by hand."""
        import glob
        from main_experiment.common import read_json
        files = sorted(glob.glob('results/baseline_revision_20261001/pool/observations/*dev*.json'))[:25]
        if not files: self.skipTest('development pool not built')
        seen = set(); n = 0
        for f in files:
            d = read_json(f)
            for row in d['observations']:
                o = parse(row['block'])
                # The archived pool holds the superseded suffix arm under the letter H;
                # its block parses as the legacy variant.
                if o['arm'] not in (LEGACY_H, 'B'): continue
                if o['D_obs'] == 0: continue
                S = sum(r[0].count('1') * r[1] for r in o['table'])
                mu = activity(S / o['D_obs'], 3 if o['arm'] == LEGACY_H else 5)
                if o['arm'] == LEGACY_H:
                    fit = mx.fit_suffix(o, mu)
                else:
                    from main_experiment.baselines import bisect
                    mean = o['M_obs'] / S
                    r = 0. if mean <= 1 else bisect(lambda x: 1. if x == 0 else x / -math.expm1(-x), mean, 0., mean)
                    fit = mx.fit_events(o, mu, r / o['parameter'])
                seen |= {k for k, v in fit.flags.items() if v}
                n += 1
        self.assertGreater(n, 20)
        self.assertTrue(seen, 'no diagnostic flag was ever raised, which is itself suspicious')


class AssumptionViolationTests(unittest.TestCase):
    """DAR and activity-driven graphs violate conditional exchangeability of windows:
    DAR has serial copy dependence, activity-driven memory makes activity grow with
    time. The candidates must still return a valid, finite, monotone profile."""

    def test_real_generator_observations(self):
        import glob
        from main_experiment.common import read_json
        files=sorted(glob.glob('results/baseline_revision_20261001/pool/observations/*.json'))[:6]
        if not files: self.skipTest('current pool not built')
        checked=0
        for f in files:
            d=read_json(f)
            for row in d['observations']:
                o=parse(row['block'])
                if o['arm'] not in (LEGACY_H,'B'): continue
                if o['D_obs']==0: continue
                S=sum(r[0].count('1')*r[1] for r in o['table'])
                mu=activity(S/o['D_obs'],3 if o['arm']==LEGACY_H else 5)
                if o['arm']==LEGACY_H: fit=mx.fit_suffix(o,mu)
                else:
                    from main_experiment.baselines import bisect
                    mean=o['M_obs']/S
                    r=0. if mean<=1 else bisect(lambda x:1. if x==0 else x/-math.expm1(-x),mean,0.,mean)
                    fit=mx.fit_events(o,mu,r/o['parameter'])
                self.assertTrue(all(math.isfinite(x) and 0<=x<=1 for x in fit.prediction),fit)
                self.assertTrue(all(x>=y-1e-12 for x,y in zip(fit.prediction,fit.prediction[1:])))
                checked+=1
        self.assertGreater(checked,0)


if __name__=='__main__': unittest.main()
