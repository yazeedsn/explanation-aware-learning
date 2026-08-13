"""
Sends a chest X-ray image to the /predict endpoint and renders the
result: the returned heatmap overlay image, plus the predicted class and
probability as the plot title.

Usage:
    python client/predict.py chest_xray.png
    python client/predict.py chest_xray.png --url http://localhost:8000/predict
    python client/predict.py chest_xray.png --save result.png --no-show
"""

import base64
import io
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import typer
from PIL import Image


def main(
    image_path: Path = typer.Argument(..., help="Path to the chest X-ray image to send."),
    url: str = typer.Option("http://localhost:8000/predict", help="The API's /predict endpoint."),
    save: Path = typer.Option(None, help="If set, save the rendered figure here instead of/as well as showing it."),
    show: bool = typer.Option(True, help="Display the figure in a window."),
    timeout: float = typer.Option(30.0, help="Request timeout in seconds."),
):
    if not image_path.exists():
        raise typer.BadParameter(f"{image_path} does not exist")

    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/png")}
        response = httpx.post(url, files=files, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

    result = response.json()
    heatmap_bytes = base64.b64decode(result["heatmap_png_base64"])
    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

    print(f"Disease:          {result['disease']}")
    print(f"Predicted class:  {result['predicted_class']}")
    print(f"Probability:      {result['probability']:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap_image)
    ax.set_title(f"{result['disease']}: {result['predicted_class']} (p={result['probability']:.3f})")
    ax.axis("off")
    fig.tight_layout()

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")

    if show:
        plt.show()


if __name__ == "__main__":
    typer.run(main)
