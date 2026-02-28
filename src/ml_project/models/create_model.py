from sklearn.ensemble import RandomForestRegressor

def create_model(model_params, x_train, y_train):
    """
    Build a regression model using values from model_params.
    Returns a scikit-learn model object.
    """
    # 1) Get the model we want (default to random_forest if missing)
    model_type = model_params.get("model_type", "random_forest")

    # 2) If it's random forest, build a RandomForestRegressor using params from YAML
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators"),
            max_depth=model_params.get("max_depth"),
            random_state=model_params.get("random_state"),
            n_jobs=model_params.get("n_jobs"),
        ).fit(x_train, y_train)

    # 3) If someone wrote an unknown model_type in yaml, fail with a clear error
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Supported options: ['random_forest']"
        )

    # 4) Return the model object
    return model