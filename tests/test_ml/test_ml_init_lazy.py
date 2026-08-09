"""Tests for lazy-loading logic in quantspt.ml.__init__."""

from __future__ import annotations

import importlib

import pytest


class TestLazyLoadingGetattr:
    """Exercise every __getattr__ branch in quantspt.ml."""

    def test_neural_fgp_lazy(self) -> None:
        from quantspt.ml import NeuralFGP
        from quantspt.ml.neural_fgp import NeuralFGP as DirectNeuralFGP

        assert NeuralFGP is DirectNeuralFGP

    def test_neural_fgp_config_lazy(self) -> None:
        from quantspt.ml import NeuralFGPConfig
        from quantspt.ml.neural_fgp import NeuralFGPConfig as DirectConfig

        assert NeuralFGPConfig is DirectConfig

    def test_input_convex_nn_lazy(self) -> None:
        from quantspt.ml import InputConvexNN
        from quantspt.ml.neural_fgp import InputConvexNN as DirectICNN

        assert InputConvexNN is DirectICNN

    def test_hmm_regime_detector_lazy(self) -> None:
        from quantspt.ml import HMMRegimeDetector
        from quantspt.ml.regime import HMMRegimeDetector as DirectHMM

        assert HMMRegimeDetector is DirectHMM

    def test_changepoint_detector_lazy(self) -> None:
        from quantspt.ml import ChangepointDetector
        from quantspt.ml.regime import ChangepointDetector as DirectCP

        assert ChangepointDetector is DirectCP

    def test_factor_model_estimator_lazy(self) -> None:
        from quantspt.ml import FactorModelEstimator
        from quantspt.ml.covariance import FactorModelEstimator as DirectFM

        assert FactorModelEstimator is DirectFM

    def test_rmt_denoiser_lazy(self) -> None:
        from quantspt.ml import RMTDenoiser
        from quantspt.ml.covariance import RMTDenoiser as DirectRMT

        assert RMTDenoiser is DirectRMT

    @pytest.mark.parametrize(
        "name",
        [
            "relative_return_loss",
            "weight_regularization",
            "turnover_penalty",
            "sharpe_of_relative_loss",
            "default_loss",
            "drift_integral_loss",
            "DriftIntegralLoss",
        ],
    )
    def test_loss_lazy_loading(self, name: str) -> None:
        import quantspt.ml
        from quantspt.ml import losses

        lazy_obj = getattr(quantspt.ml, name)
        direct_obj = getattr(losses, name)
        assert lazy_obj is direct_obj

    def test_nonexistent_attr_raises(self) -> None:
        import quantspt.ml

        with pytest.raises(AttributeError, match="no attribute"):
            _ = quantspt.ml.this_does_not_exist

    def test_getattr_via_module_reload(self) -> None:
        """Reload the module and verify __getattr__ still works."""
        import quantspt.ml

        importlib.reload(quantspt.ml)
        obj = quantspt.ml.NeuralFGP
        assert obj is not None
