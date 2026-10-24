import importlib
import numpy as np

train_module = importlib.import_module("examples.3_train_model")
build_model = train_module.build_model
create_synthetic_data = train_module.create_synthetic_data


def test_build_model():
    model = build_model(input_dim=16, num_classes=4)
    assert model.name == "mlx_mlp_classifier"
    assert len(model.layers) == 3


def test_create_synthetic_data():
    x, y = create_synthetic_data(num_samples=20, input_dim=16, num_classes=4)
    assert x.shape == (20, 16)
    assert y.shape == (20, 4)
    assert np.allclose(np.sum(y, axis=-1), 1.0)


def test_fit_and_predict_step():
    model = build_model(input_dim=16, num_classes=4)
    x, y = create_synthetic_data(num_samples=32, input_dim=16, num_classes=4)
    history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
    assert "loss" in history.history
    preds = model.predict(x[:2], verbose=0)
    assert preds.shape == (2, 4)
