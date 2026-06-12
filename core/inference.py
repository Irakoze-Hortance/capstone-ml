import time
import numpy as np

from models.prediction import ClassProbability

from core.config import (
    IMG_SIZE,
    CLASS_NAMES,
)

from core.model_loader import (
    MODEL_BACKEND,
    keras_model,
    tflite_interpreter,
    tflite_input_idx,
    tflite_output_idx,
    tflite_input_dtype,
)

from core.preprocessing import (
    preprocess_for_cnn,
    preprocess_for_uint8,
)


def infer(image):

    start = time.perf_counter()

    if MODEL_BACKEND == "tflite-int8":

        if tflite_input_dtype == np.uint8:
            batch = preprocess_for_uint8(image)

        else:
            batch = (
                preprocess_for_cnn(image)
                * 255
            ).astype(np.uint8)

        tflite_interpreter.set_tensor(
            tflite_input_idx,
            batch,
        )

        tflite_interpreter.invoke()

        raw = tflite_interpreter.get_tensor(
            tflite_output_idx
        )[0]

        probs = (
            raw.astype(np.float32)
            / 255.0
        )

    elif MODEL_BACKEND == "keras":

        batch = preprocess_for_cnn(image)

        probs = keras_model.predict(
            batch,
            verbose=0,
        )[0]

    else:

        arr = (
            np.asarray(
                image.resize(IMG_SIZE),
                dtype=np.float32,
            )
            / 255.0
        )

        brightness = float(arr.mean())

        authentic = np.clip(
            brightness * 1.2,
            0,
            1,
        )

        counterfeit = np.clip(
            1 - brightness * 1.2,
            0,
            1,
        )

        probs = np.array(
            [
                authentic,
                counterfeit,
            ],
            dtype=np.float32,
        )

        probs /= probs.sum()

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    auth_prob = float(
        probs[
            CLASS_NAMES.index(
                "authentic"
            )
        ]
    )

    counterfeit_prob = float(
        probs[
            CLASS_NAMES.index(
                "counterfeit"
            )
        ]
    )

    winner_idx = int(
        np.argmax(probs)
    )

    verdict = CLASS_NAMES[winner_idx]

    confidence = float(
        probs[winner_idx]
    )

    return (
        verdict,
        confidence,
        ClassProbability(
            authentic=auth_prob,
            counterfeit=counterfeit_prob,
        ),
        inference_ms,
    )