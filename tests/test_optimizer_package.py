def test_optimizer_package_keeps_public_api_lazy_and_available():
    from optimizer import OptimizationRequest, Optimizer

    assert Optimizer is not None
    assert OptimizationRequest is not None
