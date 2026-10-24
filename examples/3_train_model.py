"""End-to-End Keras Model Training and Inference with MLX Backend.

Builds, compiles, and trains a multi-layer perceptron on synthetic data,
verifying forward pass, backpropagation, and weight updates on Apple Silicon GPU.
"""

import os
import numpy as np

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402


def build_model(input_dim=64, num_classes=10):
    """Builds and compiles an MLP classifier."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlx_mlp_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_synthetic_data(num_samples=1000, input_dim=64, num_classes=10, seed=42):
    """Generates synthetic classification dataset."""
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, input_dim).astype("float32")
    y_indices = rng.randint(0, num_classes, size=(num_samples,))
    y = np.zeros((num_samples, num_classes), dtype="float32")
    y[np.arange(num_samples), y_indices] = 1.0
    return x, y


def main():
    print("=" * 60)
    print(f"Training Keras Model with Backend: {keras.config.backend()}")
    print("=" * 60)

    model = build_model()
    model.summary()

    x_train, y_train = create_synthetic_data(num_samples=1000)

    print("\nTraining model...")
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )

    print("\nRunning inference...")
    x_test = x_train[:5]
    predictions = model.predict(x_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted class probabilities sum: {np.sum(predictions, axis=-1)}")
    print(f"Predicted classes: {np.argmax(predictions, axis=-1)}")
    print("\nTraining & inference finished successfully with Keras MLX backend!")


if __name__ == "__main__":
    main()
