"""Tests for contrib/__init__.py -- plugin registration and discovery."""

from __future__ import annotations

from quantspt.contrib import (
    _MODEL_REGISTRY,
    _PORTFOLIO_REGISTRY,
    _PROVIDER_REGISTRY,
    discover_models,
    discover_portfolios,
    discover_providers,
    list_models,
    list_portfolios,
    list_providers,
    register_data_provider,
    register_model,
    register_portfolio,
)


class _RegistryCleanup:
    """Mixin that saves and restores registry state around each test."""

    def setup_method(self) -> None:
        self._saved_providers = dict(_PROVIDER_REGISTRY)
        self._saved_portfolios = dict(_PORTFOLIO_REGISTRY)
        self._saved_models = dict(_MODEL_REGISTRY)

    def teardown_method(self) -> None:
        _PROVIDER_REGISTRY.clear()
        _PROVIDER_REGISTRY.update(self._saved_providers)
        _PORTFOLIO_REGISTRY.clear()
        _PORTFOLIO_REGISTRY.update(self._saved_portfolios)
        _MODEL_REGISTRY.clear()
        _MODEL_REGISTRY.update(self._saved_models)


class TestRegisterDataProvider(_RegistryCleanup):
    def test_decorator_registers(self) -> None:
        @register_data_provider("test_prov")
        class MyProv:
            pass

        assert "test_prov" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["test_prov"] is MyProv

    def test_decorator_returns_class(self) -> None:
        @register_data_provider("another")
        class AnotherProv:
            pass

        assert AnotherProv.__name__ == "AnotherProv"


class TestRegisterPortfolio(_RegistryCleanup):
    def test_decorator_registers(self) -> None:
        @register_portfolio("test_port")
        class MyPort:
            pass

        assert "test_port" in _PORTFOLIO_REGISTRY

    def test_decorator_returns_class(self) -> None:
        @register_portfolio("p2")
        class Port2:
            pass

        assert Port2.__name__ == "Port2"


class TestRegisterModel(_RegistryCleanup):
    def test_decorator_registers(self) -> None:
        @register_model("test_model")
        class MyModel:
            pass

        assert "test_model" in _MODEL_REGISTRY

    def test_decorator_returns_class(self) -> None:
        @register_model("m2")
        class Model2:
            pass

        assert Model2.__name__ == "Model2"


class TestDiscovery(_RegistryCleanup):
    def test_discover_providers_includes_runtime(self) -> None:
        @register_data_provider("rt_prov")
        class RtProv:
            pass

        result = discover_providers()
        assert "rt_prov" in result
        assert result["rt_prov"] is RtProv

    def test_discover_portfolios_includes_runtime(self) -> None:
        @register_portfolio("rt_port")
        class RtPort:
            pass

        result = discover_portfolios()
        assert "rt_port" in result

    def test_discover_models_includes_runtime(self) -> None:
        @register_model("rt_model")
        class RtModel:
            pass

        result = discover_models()
        assert "rt_model" in result

    def test_list_providers_returns_set(self) -> None:
        @register_data_provider("lp")
        class LP:
            pass

        result = list_providers()
        assert isinstance(result, set)
        assert "lp" in result

    def test_list_portfolios_returns_set(self) -> None:
        @register_portfolio("lport")
        class LPort:
            pass

        result = list_portfolios()
        assert isinstance(result, set)
        assert "lport" in result

    def test_list_models_returns_set(self) -> None:
        @register_model("lm")
        class LM:
            pass

        result = list_models()
        assert isinstance(result, set)
        assert "lm" in result

    def test_empty_registries(self) -> None:
        _PROVIDER_REGISTRY.clear()
        _PORTFOLIO_REGISTRY.clear()
        _MODEL_REGISTRY.clear()
        assert isinstance(discover_providers(), dict)
        assert isinstance(discover_portfolios(), dict)
        assert isinstance(discover_models(), dict)
