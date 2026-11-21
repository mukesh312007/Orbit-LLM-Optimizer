from config.model_registry import load_model_registry


def test_registry_identifiers_are_unique():
    models = load_model_registry()
    assert models and len({x["model_id"] for x in models}) == len(models)
