"""Independent density checks for the Student-t scale-mixture implementation."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma, norm, multivariate_normal
from analysis.robust_model import point_loglik

class DensityChecks(unittest.TestCase):
    def test_t_density_equals_integrated_normal_gamma_mixture(self):
        y=np.array([.4,-.2,.7]); X=np.ones((3,1)); sigma=.3; phi=.4
        draws=np.array([[.1,sigma,phi]])
        residual=y-.1
        for nu in [3,5,10]:
            expected=[]
            # Independently integrate the latent scale at each observation.
            for j in range(3):
                location=0 if j==0 else phi*residual[j-1]
                scale=sigma/np.sqrt(1-phi**2) if j==0 else sigma
                density=quad(lambda w:norm.pdf(residual[j],loc=location,scale=scale/np.sqrt(w))*gamma.pdf(w,nu/2,scale=2/nu),
                             0,np.inf,epsabs=1e-11)[0]
                expected.append(np.log(density))
            np.testing.assert_allclose(point_loglik(y,X,draws,'student',nu)[0],expected,atol=1e-8)

    def test_gaussian_likelihood_equals_joint_ar_covariance(self):
        y=np.array([.4,-.2,.7]); X=np.ones((3,1)); draws=np.array([[.1,.3,.4]])
        k=np.arange(3); cov=.3**2/(1-.4**2)*.4**np.abs(k[:,None]-k[None,:])
        expected=multivariate_normal.logpdf(y,mean=np.repeat(.1,3),cov=cov)
        self.assertAlmostEqual(point_loglik(y,X,draws,'gaussian').sum(),expected,places=10)

if __name__=='__main__': unittest.main()
