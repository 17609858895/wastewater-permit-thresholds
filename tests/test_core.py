import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hicboost import Family,nll_and_grads,CrossFitHIC
import cliffmix
spec=importlib.util.spec_from_file_location('bounds_module',Path(__file__).resolve().parents[1]/'scripts/05_fit_counterfactual.py')
bounds_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounds_module)

class CoreTests(unittest.TestCase):
    def test_interval_gradients(self):
        mu=np.array([.1,.3,-.2]);s=np.array([-.2,.1,.4]);lo=np.array([-.7,-np.inf,.2]);hi=np.array([.8,.5,np.inf]);eps=1e-6
        for name in ['normal','logistic']:
            family=Family(name);result=nll_and_grads(family,mu,s,lo,hi)
            dm=(nll_and_grads(family,mu+eps,s,lo,hi)[0]-nll_and_grads(family,mu-eps,s,lo,hi)[0])/(2*eps)
            ds=(nll_and_grads(family,mu,s+eps,lo,hi)[0]-nll_and_grads(family,mu,s-eps,lo,hi)[0])/(2*eps)
            np.testing.assert_allclose(result[1],dm,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(result[2],ds,rtol=1e-5,atol=1e-6)
    def test_nondetect_precedes_window(self):
        d=pd.DataFrame({'v':[9.,9.,0.,25.],'h':[.5]*4,'nd':[0,1,0,0]})
        lo,hi,_=bounds_module.bounds(d,.8,np.full(4,10.),2.)
        np.testing.assert_allclose([lo[0],hi[0]],np.log([8.,20.]))
        self.assertTrue(np.isneginf(lo[1]));self.assertAlmostEqual(hi[1],np.log(9.))
        self.assertTrue(np.isneginf(lo[2]));self.assertAlmostEqual(hi[2],np.log(.5))
        np.testing.assert_allclose([lo[3],hi[3]],np.log([24.5,25.5]))
    def test_hybrid_count(self):
        d=pd.DataFrame({'T':[10.5]*5,'L':[10.]*5,'v':[10.,11.,20.,25.,11.],'nd':[0,0,0,0,1], 'mu_n':[2.]*5,'s_n':[0.]*5,'mu_l':[2.]*5,'s_l':[0.]*5})
        q=cliffmix.window_quantities(d,{'w_normal':.2})
        self.assertEqual(q.O_win.sum(),1);self.assertEqual(q.O_abv.sum(),2)
        expected=cliffmix.cdf(np.log(20.),2.,0.,2.,0.,.2)-cliffmix.cdf(np.log(10.5),2.,0.,2.,0.,.2)
        np.testing.assert_allclose(q.E_win,expected)
    def test_small_permit_grouped_fit(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(360,3));y=.3*X[:,0]+rng.normal(0,.3,360);groups=np.repeat(np.arange(60),6)
        for family in ['normal','logistic']:
            model=CrossFitHIC(k=3,family=family,seed=0,cycles=2,rounds_per_cycle=2,min_data_in_leaf=20,n_jobs=1).fit(X,y-.05,y+.05,groups=groups,recalibrate=False)
            self.assertTrue(np.isfinite(model.mu_oof).all());self.assertTrue(np.isfinite(model.s_oof).all())
            self.assertEqual(set(model.fold),{0,1,2})
            self.assertTrue(all(len(set(model.fold[groups==g]))==1 for g in np.unique(groups)))

if __name__=='__main__':unittest.main()
