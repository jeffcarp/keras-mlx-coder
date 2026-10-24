import os
import mlx.core as mx

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402


def test_keras_backend_is_mlx():
    assert keras.config.backend() == "mlx"


def test_tensor_is_mlx_array():
    tensor = keras.random.normal(shape=(3, 3))
    assert isinstance(tensor, mx.array)
    out = keras.ops.matmul(tensor, tensor)
    mx.eval(out)
    assert out.shape == (3, 3)
