"""Run automated end-to-end inference verification across all sample images."""

import os
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app


def test_samples():
    print("=" * 60)
    print("OBJECT DETECTOR & COUNTER - END-TO-END VERIFICATION")
    print("=" * 60)

    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "sample_images")
    samples = [
        ("street.jpg", "standard", 0.35, None),
        ("workspace.jpg", "standard", 0.35, None),
        ("living_room.jpg", "standard", 0.35, None),
        ("workspace.jpg", "open_vocab", 0.15, "laptop, keyboard, mouse, monitor, table"),
    ]

    with TestClient(app) as client:
        # 1. Health check
        health_resp = client.get("/health")
        print("\n[1] Health Check:")
        print(f"Status Code: {health_resp.status_code}")
        print(f"Response: {health_resp.json()}")
        assert health_resp.status_code == 200

        # 2. Iterate each sample image
        for fname, mode, threshold, custom_labels in samples:
            img_path = os.path.join(sample_dir, fname)
            print(f"\n[2] Testing Image: {fname} (Mode: {mode}, Threshold: {threshold})")

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            data = {
                "threshold": str(threshold),
                "nms_threshold": "0.50",
                "mode": mode,
            }
            if custom_labels:
                data["custom_labels"] = custom_labels

            resp = client.post(
                "/detect",
                files={"file": (fname, img_bytes, "image/jpeg")},
                data=data,
            )

            print(f"Status Code: {resp.status_code}")
            assert resp.status_code == 200, f"Failed with {resp.text}"

            result = resp.json()
            total = result["total"]
            counts = result["counts"]
            detections = result["detections"]
            has_b64 = bool(result["annotated_image"])

            print(f"  -> Total Objects Detected: {total}")
            print(f"  -> Per-Class Breakdown: {counts}")
            print(f"  -> Annotated Image Generated: {has_b64} (length: {len(result['annotated_image'])} chars)")
            print("  -> First 3 Detections:")
            for d in detections[:3]:
                print(f"     * {d['label']} ({d['confidence'] * 100:.1f}%) box: {d['box']}")

    print("\n" + "=" * 60)
    print("ALL SAMPLE IMAGE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_samples()

