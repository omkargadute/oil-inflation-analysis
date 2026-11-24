import unittest
import numpy as np
from scipy.stats import t,norm,halfnorm
from analysis.extension_model import ar_terms,conditional_loglike,INITIAL
from analysis.compiled_sampler import coefficients,target

class ExtensionChecks(unittest.TestCase):
    def test_stability_coordinates(self):
        rng=np.random.default_rng(5)
        for kind in ['ar1','ar2','sar12']:
            for _ in range(30):
                k=rng.uniform(-.98,.98,1 if kind=='ar1' else 2);lags,a=ar_terms(kind,k)
                polynomial=np.zeros(max(lags)+1);polynomial[0]=1
                for lag,value in zip(lags,a):polynomial[lag]=-value
                self.assertLess(np.max(np.abs(np.roots(polynomial))),1)
                np.testing.assert_allclose(a,coefficients({'ar1':1,'ar2':2,'sar12':12}[kind],k))

    def test_seasonal_conditional_density(self):
        # Direct application of the two lag polynomials is independent of ar_terms.
        y=np.linspace(-.5,.8,30)+np.sin(np.arange(30));X=np.ones((30,1));b=np.array([.2])
        r=y-.2;phi,Phi=.4,-.3
        e=np.array([r[j]-phi*r[j-1]-Phi*r[j-12]+phi*Phi*r[j-13] for j in range(INITIAL,len(y))])
        expected=t.logpdf(e/.3,df=5)-np.log(.3)
        np.testing.assert_allclose(conditional_loglike(y,X,b,.3,[phi,Phi],'sar12'),expected,atol=1e-12)

    def test_compiled_scale_and_AR_target_against_normal_density(self):
        r=np.cos(np.arange(40));cur=r[INITIAL:]
        R=np.column_stack([r[INITIAL-lag:len(r)-lag] for lag in [1,12,13]])
        rr=cur@cur;h=R.T@cur;G=R.T@R
        states=[(np.log(.3),np.array([.2,-.3])),(np.log(.5),np.array([.5,.1]))]
        actual=[];expected=[]
        for ls,eta in states:
            k=np.tanh(eta);phi,Phi=k
            e=cur-phi*R[:,0]-Phi*R[:,1]+phi*Phi*R[:,2]
            expected.append(norm.logpdf(e,scale=np.exp(ls)).sum()+halfnorm.logpdf(np.exp(ls),scale=.5)+ls+np.log1p(-k*k).sum())
            actual.append(target(ls,eta,12,len(cur),rr,h,G))
        self.assertAlmostEqual(actual[0]-actual[1],expected[0]-expected[1],places=9)

    def test_forecast_does_not_use_current_or_future_outcome(self):
        import pandas as pd
        from analysis.robust_model import specification
        from analysis.extension_model import forecast_extension
        frame=pd.DataFrame({'date':pd.date_range('2000-01-01',periods=40,freq='MS'),
                            'month':np.tile(np.arange(1,13),4)[:40],
                            'T':np.zeros(40),'C':np.zeros(40),'monthly':np.zeros(40)})
        X,names,_=specification(frame)
        state=np.r_[np.zeros(X.shape[1]),.2,.4]
        fit={'chains':np.tile(state,(4,10,1)),'names':names+['sigma','kappa1'],
             'adjusted':True,'oil':True,'kind':'ar1','nu':5}
        first=forecast_extension(fit,frame,25)['prediction']
        changed=frame.copy();changed.loc[25,'monthly']=10
        second=forecast_extension(fit,changed,25)['prediction']
        self.assertEqual(first[0],second[0])
        self.assertAlmostEqual(second[1]-first[1],4)
        np.testing.assert_allclose(first[2:],second[2:])

    def test_paired_block_comparison_is_zero_for_identical_errors(self):
        from analysis.strengthening import block_gain,block_indices
        errors=np.linspace(-1,1,60)
        np.testing.assert_allclose(block_gain(errors,errors,seed=42),[0,0])
        index=block_indices(60,42)
        self.assertEqual(index.shape,(2000,60))
        self.assertTrue(np.all((index>=0)&(index<60)))

if __name__=='__main__':unittest.main()
