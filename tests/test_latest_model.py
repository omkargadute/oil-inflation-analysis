import unittest
import numpy as np
from analysis.latest_data import prepare_latest
from analysis.latest_model import observed_indices,forecast_latest
from analysis.latest_sampler import draw_observed_chain
from analysis.compiled_sampler import draw_chain
from analysis.robust_model import specification

class LatestChecks(unittest.TestCase):
    def test_missing_month_is_not_bridged(self):
        panel,*_=prepare_latest()
        g=panel[panel.iso3.eq('USA')].reset_index(drop=True)
        lookup=g.set_index('date')
        self.assertTrue(np.isnan(lookup.loc['2025-10-01','cpi']))
        self.assertTrue(np.isnan(lookup.loc['2025-11-01','monthly']))
        self.assertTrue(np.isfinite(lookup.loc['2025-12-01','monthly']))
        dates=set(g.date.iloc[observed_indices(g)].dt.strftime('%Y-%m'))
        self.assertNotIn('2025-10',dates);self.assertNotIn('2025-11',dates)
        self.assertNotIn('2025-12',dates);self.assertIn('2026-01',dates)

    def test_specialized_sampler_equals_complete_calendar_kernel(self):
        panel,*_=prepare_latest()
        g=panel[panel.iso3.eq('CAN')].iloc[:100]
        X,_,scales=specification(g);y=g.monthly.to_numpy();indices=observed_indices(g)
        a,_=draw_chain(np.ascontiguousarray(X),y,scales,1,5,123,20,40)
        b,_=draw_observed_chain(np.ascontiguousarray(X),y,scales,indices,5,123,20,40)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)

    def test_recent_forecast_preserves_missing_calendar_exclusions(self):
        panel,*_=prepare_latest();g=panel[panel.iso3.eq('USA')].reset_index(drop=True)
        X,names,_=specification(g);state=np.r_[np.zeros(X.shape[1]),.2,.4]
        fit={'chains':np.tile(state,(4,10,1)),'names':names+['sigma','kappa1'],'oil':True}
        forecast=forecast_latest(fit,g)
        excluded=forecast[~forecast.eligible].date.dt.strftime('%Y-%m').tolist()
        self.assertEqual(excluded,['2025-10','2025-11','2025-12'])
        self.assertTrue(forecast[forecast.date.ge('2026-01-01')].eligible.all())

if __name__=='__main__':unittest.main()
