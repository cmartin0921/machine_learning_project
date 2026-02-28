def train_model(model, x_train, y_train):
    """Fit the model on training data."""
    model.fit(x_train, y_train)
    return model
