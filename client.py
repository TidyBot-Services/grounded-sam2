"""Grounded SAM 2 Client SDK — no external dependencies."""

import base64
import json
import urllib.request
import urllib.error


class Client:
    """Client for the Grounded SAM 2 open-vocabulary detection + segmentation service.

    Example:
        client = Client("http://server:8001")
        assert client.health()
        with open("photo.jpg", "rb") as f:
            result = client.predict(f.read(), ["cat", "dog"])
        for det in result["detections"]:
            print(det["label"], det["confidence"], det["box"])
    """

    def __init__(self, host: str = "http://localhost:8001") -> None:
        """Initialize the client.

        Args:
            host: Base URL of the Grounded SAM 2 service.
        """
        self.host = host.rstrip("/")

    def health(self) -> bool:
        """Check if the service is reachable.

        Returns:
            True if the service responds to /health, False otherwise.

        Example:
            >>> client = Client("http://localhost:8001")
            >>> client.health()
            True
        """
        try:
            req = urllib.request.Request(f"{self.host}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def predict(
        self,
        image: bytes,
        prompts: list[str],
        *,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> dict:
        """Run open-vocabulary detection + segmentation on an image.

        Args:
            image: Raw image bytes (JPEG or PNG).
            prompts: List of text prompts, e.g. ["cat", "dog"].
            box_threshold: Confidence threshold for bounding box detection.
            text_threshold: Confidence threshold for text-prompt matching.

        Returns:
            dict with keys:
                - detections: list of dicts with label, confidence, box, mask_rle
                - width: image width in pixels
                - height: image height in pixels

        Example:
            >>> client = Client("http://localhost:8001")
            >>> with open("image.jpg", "rb") as f:
            ...     result = client.predict(f.read(), ["person", "chair"])
            >>> for d in result["detections"]:
            ...     print(d["label"], d["box"])
        """
        payload = json.dumps({
            "image": base64.b64encode(image).decode(),
            "prompts": prompts,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def predict_masks_as_bytes(
        self,
        image: bytes,
        prompts: list[str],
        **kwargs,
    ) -> list[dict]:
        """Like predict(), but decodes mask PNGs to bytes for convenience.

        Args:
            image: Raw image bytes (JPEG or PNG).
            prompts: Text prompts for detection.
            **kwargs: Passed to predict().

        Returns:
            List of dicts with label, confidence, box, mask_png (bytes).

        Example:
            >>> results = client.predict_masks_as_bytes(img_bytes, ["cup"])
            >>> with open("mask.png", "wb") as f:
            ...     f.write(results[0]["mask_png"])
        """
        resp = self.predict(image, prompts, **kwargs)
        out = []
        for det in resp["detections"]:
            out.append({
                "label": det["label"],
                "confidence": det["confidence"],
                "box": det["box"],
                "mask_png": base64.b64decode(det["mask_rle"]),
            })
        return out
