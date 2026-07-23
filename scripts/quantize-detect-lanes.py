#!/usr/bin/env python3
"""QDQ static int8 quantization for the browser detection lanes (wasm floor).

Why QDQ: onnxruntime-web's WASM EP has no ConvInteger kernel, so
quantize_dynamic (QOperator) output cannot load there. QDQ emits
QuantizeLinear/DequantizeLinear + QLinearConv, which the wasm build runs.

The detect head (final /model.N/ module) is EXCLUDED: quantized heads collapse
(measured: zero detections whole-graph; hallucinated labels head-included).
Backbone+neck int8 keeps parity — YOLO-World s: score drift ≤0.05, IoU ≥0.99
vs fp32. YOLOE int8 was evaluated and REJECTED (label hallucinations persist
even with the head fp32) — quantize the yolo-world lane only.

Calibration: photos from CALIB_DIR through the exact browser preprocessing
(640 top-left letterbox, gray-128 pad, RGB/255); YOLO-World txt_feats are
calibrated with real rows from the shipped clip-vocab table (1-4 random words
padded by repetition to the 32 slots, mirroring clip-text.embedSlots).

Usage:
  CALIB_DIR=~/photos python scripts/quantize-detect-lanes.py yolo-world:s yolo-world:m
  (default jobs: yolo-world:s; CALIB_DIR needs ~20-40 varied JPEGs)
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CALIB = Path(os.environ.get("CALIB_DIR", ROOT / "calib")).expanduser()
SIDE, PAD, SLOTS, DIM = 640, 128, 32, 512
random.seed(7)


def build_frame(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = SIDE / max(w, h)
    resized = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    canvas = Image.new("RGB", (SIDE, SIDE), (PAD, PAD, PAD))
    canvas.paste(resized, (0, 0))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def clip_rows():
    meta = json.loads((ROOT / "models/clip-vocab/clip-vocab.json").read_text())
    return np.fromfile(ROOT / "models/clip-vocab/clip-vocab.f32", dtype="<f4").reshape(meta["count"], meta["dim"])


class FrameReader(CalibrationDataReader):
    def __init__(self, with_txt):
        paths = sorted([*CALIB.glob("*.jpg"), *CALIB.glob("*.jpeg"), *CALIB.glob("*.png")])
        if len(paths) < 8:
            raise SystemExit(f"CALIB_DIR={CALIB} has {len(paths)} images; need ≥8 varied photos")
        self.frames = [build_frame(p) for p in paths]
        self.table = clip_rows() if with_txt else None
        self.i = 0

    def get_next(self):
        if self.i >= len(self.frames):
            return None
        feed = {"images": self.frames[self.i]}
        if self.table is not None:
            picks = [random.randrange(len(self.table)) for _ in range(random.randint(1, 4))]
            feats = np.zeros((1, SLOTS, DIM), dtype=np.float32)
            for s in range(SLOTS):
                feats[0, s] = self.table[picks[s % len(picks)]]
            feed["txt_feats"] = feats
        self.i += 1
        return feed


def quantize(lane, scale):
    if lane != "yolo-world":
        raise SystemExit(f"lane {lane}: only yolo-world is quantized (YOLOE int8 rejected on quality)")
    src = ROOT / f"models/yolo-world/yolo-world-{scale}.onnx"
    dst = ROOT / f"models/yolo-world/yolo-world-{scale}.int8.onnx"
    prep = dst.with_suffix(".prep.onnx")
    print(f"[quant] {lane}:{scale} pre-process…", flush=True)
    # skip_symbolic_shape: TopK-style heads break symbolic_shape_infer
    quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
    g = onnx.load(str(prep), load_external_data=False).graph
    head_idx = max(int(m.group(1)) for n in g.node for m in [re.match(r"/model\.(\d+)/", n.name)] if m)
    exclude = [n.name for n in g.node if n.name.startswith(f"/model.{head_idx}/")]
    print(f"[quant] {lane}:{scale} calibrate+quantize (head /model.{head_idx}/ excluded, {len(exclude)} nodes)…", flush=True)
    quantize_static(
        str(prep), str(dst), FrameReader(with_txt=True),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)
    print(f"[quant] {lane}:{scale} -> {dst.name} {dst.stat().st_size/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["yolo-world:s"]
    for job in jobs:
        lane, scale = job.split(":")
        quantize(lane, scale)
    print("[quant] done")
