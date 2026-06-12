import numpy as np

from core.config import (
    MODEL_KERAS,
    MODEL_TFLITE,
    log,
)

MODEL_BACKEND = "none"
MODEL_VERSION = "unavailable"

tflite_interpreter = None
tflite_input_idx = 0
tflite_output_idx = 0
tflite_input_dtype = None

keras_model = None


def load_tflite():
    global tflite_interpreter, tflite_input_idx, tflite_output_idx, tflite_input_dtype

    if not MODEL_TFLITE.exists():
        return False

    try:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(
            model_path=str(MODEL_TFLITE)
        )

    except ImportError:

        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(
                model_path=str(MODEL_TFLITE)
            )

        except Exception as exc:
            log.warning("TFLite load failed: %s", exc)
            return False

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    tflite_interpreter = interpreter
    tflite_input_idx = inp["index"]
    tflite_output_idx = out["index"]
    tflite_input_dtype = inp["dtype"]

    log.info(
        "Loaded TFLite model: %s",
        MODEL_TFLITE.name,
    )

    return True


def load_keras():
    global keras_model

    if not MODEL_KERAS.exists():
        return False

    try:
        import tensorflow as tf

        keras_model = tf.keras.models.load_model(
            str(MODEL_KERAS)
        )

        log.info(
            "Loaded keras model: %s",
            MODEL_KERAS.name,
        )

        return True

    except Exception as exc:
        log.warning(
            "Keras load failed: %s",
            exc,
        )
        return False


def initialize_model():
    global MODEL_BACKEND
    global MODEL_VERSION

    if load_tflite():
        MODEL_BACKEND = "tflite-int8"
        MODEL_VERSION = MODEL_TFLITE.stem

    elif load_keras():
        MODEL_BACKEND = "keras"
        MODEL_VERSION = MODEL_KERAS.stem

    else:
        MODEL_BACKEND = "pixel-similarity"
        MODEL_VERSION = "baseline-v1"

        log.warning(
            "No trained model found. Using fallback baseline."
        )


initialize_model()