import unittest
import numpy as np
import pandas as pd
from analysis.combined_sampler import draw_combined,latent_conditional
from analysis.compiled_sampler import draw_chain
from analysis.robust_model import specification
from analysis.combined_model import forecast_combined

class CombinedChecks(unittest.TestCase):
    def test_complete_student_kernel_is_unchanged(self):
        rng=np.random.default_rng(5)
        X=np.ascontiguousarray(np.column_stack([np.ones(50),rng.normal(size=50)]))
        y=np.ascontiguousarray(rng.normal(size=50));scales=np.array([.5,.3])
        a,_=draw_chain(X,y,scales,1,5,123,20,40)
        b,_=draw_combined(X,y,scales,1,5,123,20,40)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)

    def test_missing_state_update_equals_gaussian_conditional(self):
        n=35;phi=.4;sigma=.3;b=np.array([.2]);X=np.ones((n,1))
        y=np.sin(np.arange(n)*.3);missing=np.array([22,23])
        mean,Q=latent_conditional(y,X,b,sigma,np.ones(n-13),1,np.array([phi]),missing)
        k=n-13;index=np.arange(k)
        covariance=sigma**2*phi**np.abs(index[:,None]-index[None,:])*(1-phi**(2*(np.minimum(index[:,None],index[None,:])+1)))/(1-phi**2)
        joint_mean=.2+phi**(index+1)*(y[12]-.2)
        m=missing-13;o=np.array([j for j in index if j not in m])
        Cmo=covariance[np.ix_(m,o)];Coo=covariance[np.ix_(o,o)]
        expected_mean=joint_mean[m]+Cmo@np.linalg.solve(Coo,y[13:][o]-joint_mean[o])
        expected_cov=covariance[np.ix_(m,m)]-Cmo@np.linalg.solve(Coo,Cmo.T)
        np.testing.assert_allclose(mean,expected_mean,atol=1e-12)
        np.testing.assert_allclose(sigma**2*np.linalg.inv(Q),expected_cov,atol=1e-12)

    def test_forecasts_never_use_latent_test_outcomes(self):
        f=pd.DataFrame({'date':pd.date_range('2000-01-01',periods=60,freq='MS'),
                        'T':np.zeros(60),'C':np.zeros(60),'monthly':np.zeros(60)})
        f['month']=f.date.dt.month;f.loc[45:46,'monthly']=np.nan
        X,names,_=specification(f);p=X.shape[1]
        fit={'chains':np.tile(np.r_[np.zeros(p),.2,.4],(4,10,1)),
             'names':names+['sigma','kappa1'],'p':p,'d':1,'kind':'ar1','family':'student','nu':5,'oil':True,'adjusted':True}
        result=forecast_combined(fit,f,f.date.iloc[40])
        self.assertEqual(result.loc[~result.eligible,'date'].tolist(),f.date.iloc[45:48].tolist())
        altered=f.copy();altered.loc[48,'monthly']=10
        other=forecast_combined(fit,altered,f.date.iloc[40])
        self.assertEqual(result.prediction.iloc[8],other.prediction.iloc[8])
        self.assertAlmostEqual(other.prediction.iloc[9]-result.prediction.iloc[9],4.)

if __name__=='__main__':unittest.main()
