import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from io import BytesIO
from fastapi import HTTPException

from core.config import (
    TERRACOTTA,
    SUCCESS,
    SLATE,
    CONF_THRESHOLD,
)


def build_heatmap_png(records):

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No history found.",
        )

    verdicts = [
        "authentic",
        "counterfeit",
    ]

    matrix = {
        v: []
        for v in verdicts
    }

    for row in records:

        verdict = row.get(
            "verdict"
        )

        if verdict in matrix:
            matrix[verdict].append(
                row.get(
                    "confidence",
                    0,
                )
            )

    rows = [
        v
        for v in verdicts
        if matrix[v]
    ]

    values = np.array(
        [
            [np.mean(matrix[r])]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 3)
    )

    heatmap = ax.imshow(
        values,
        cmap="Oranges",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        "Mean Confidence by Verdict"
    )

    plt.colorbar(
        heatmap,
        ax=ax,
    )

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer.getvalue()


def build_confidence_dist_png(records):

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No history found.",
        )

    auth = [
        r["confidence"]
        for r in records
        if r["verdict"]
        == "authentic"
    ]

    fake = [
        r["confidence"]
        for r in records
        if r["verdict"]
        == "counterfeit"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bins = np.linspace(
        0,
        1,
        21,
    )

    if auth:
        ax.hist(
            auth,
            bins=bins,
            alpha=0.7,
            color=SUCCESS,
            label="Authentic",
        )

    if fake:
        ax.hist(
            fake,
            bins=bins,
            alpha=0.7,
            color=TERRACOTTA,
            label="Counterfeit",
        )

    ax.axvline(
        CONF_THRESHOLD,
        color=SLATE,
        linestyle="--",
    )

    ax.legend()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer.getvalue()