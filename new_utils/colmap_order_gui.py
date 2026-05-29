#!/usr/bin/env python3
"""Browser GUI for inspecting COLMAP cameras and manually labeling an order."""

import argparse
import json
import math
import os
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import qvec2rotmat


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COLMAP Order GUI</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d7dde5;
      --text: #1c2430;
      --muted: #667085;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --blue: #2563eb;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #app {
      display: grid;
      grid-template-columns: 1fr 360px;
      height: 100vh;
    }
    #stageWrap {
      position: relative;
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--line);
      background: #eef2f6;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      touch-action: none;
      user-select: none;
    }
    #tooltip {
      position: absolute;
      pointer-events: none;
      display: none;
      max-width: 320px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,0.96);
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-line;
    }
    aside {
      display: flex;
      flex-direction: column;
      min-width: 0;
      background: var(--panel);
    }
    .bar {
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .title {
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 4px;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    button, select, input {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font-size: 13px;
      height: 34px;
    }
    button {
      padding: 0 10px;
      cursor: pointer;
      white-space: nowrap;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.warn { color: var(--danger); }
    button:disabled { opacity: 0.45; cursor: default; }
    select { padding: 0 8px; }
    input {
      padding: 0 9px;
      min-width: 0;
      width: 100%;
    }
    .statgrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }
    .stat b {
      display: block;
      font-size: 15px;
    }
    .stat span {
      color: var(--muted);
      font-size: 12px;
    }
    #selectedList {
      min-height: 0;
      flex: 1;
      overflow: auto;
      border-top: 1px solid var(--line);
    }
    .item {
      display: grid;
      grid-template-columns: 48px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 8px 12px;
      border-bottom: 1px solid #edf0f4;
      font-size: 12px;
    }
    .item:hover { background: #f6f8fb; }
    .idx {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .mini {
      height: 26px;
      padding: 0 8px;
      font-size: 12px;
    }
    #status {
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 900px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 1fr 330px; }
      #stageWrap { border-right: 0; border-bottom: 1px solid var(--line); }
      aside { min-height: 0; }
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="stageWrap">
      <canvas id="canvas"></canvas>
      <div id="tooltip"></div>
    </div>
    <aside>
      <div class="bar">
        <div class="title">COLMAP Order GUI</div>
        <div class="sub" id="datasetPath"></div>
        <div class="statgrid">
          <div class="stat"><b id="cameraCount">0</b><span>cameras</span></div>
          <div class="stat"><b id="selectedCount">0</b><span>selected</span></div>
        </div>
        <div class="row">
          <select id="projection" title="Projection">
            <option value="xy">XY</option>
            <option value="xz">XZ</option>
            <option value="yz">YZ</option>
            <option value="pca">PCA XY</option>
          </select>
          <select id="colorMode" title="Color">
            <option value="height">Height</option>
            <option value="name">Image Name</option>
            <option value="selected">Selected</option>
          </select>
          <button id="fitBtn">Fit</button>
        </div>
        <div class="row">
          <input id="search" placeholder="Search image name">
          <button id="goBtn">Go</button>
        </div>
        <div class="row">
          <button id="undoBtn">Undo</button>
          <button id="clearBtn" class="warn">Clear</button>
          <button id="saveBtn" class="primary">Save mapping.txt</button>
        </div>
        <div id="status">Loading cameras...</div>
      </div>
      <div id="selectedList"></div>
    </aside>
  </div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const datasetPath = document.getElementById('datasetPath');
const cameraCount = document.getElementById('cameraCount');
const selectedCount = document.getElementById('selectedCount');
const selectedList = document.getElementById('selectedList');
const projectionEl = document.getElementById('projection');
const colorModeEl = document.getElementById('colorMode');
const searchEl = document.getElementById('search');
const statusEl = document.getElementById('status');

let cameras = [];
let points = [];
let selected = [];
let selectedSet = new Set();
let hover = -1;
let focusIndex = -1;
let pca = null;
let heightRange = [0, 1];
let view = { scale: 1, tx: 0, ty: 0 };
let drag = null;
let lastCanvasSize = null;

function setStatus(text) { statusEl.textContent = text; }

function resize() {
  const rect = canvas.getBoundingClientRect();
  const oldCenter = lastCanvasSize
    ? screenToWorld(lastCanvasSize.width / 2, lastCanvasSize.height / 2)
    : null;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (oldCenter && rect.width > 0 && rect.height > 0) {
    view.tx = rect.width / 2 - oldCenter[0] * view.scale;
    view.ty = rect.height / 2 + oldCenter[1] * view.scale;
  }
  lastCanvasSize = { width: rect.width, height: rect.height };
  draw();
}

function cameraPoint(cam) {
  const c = cam.center;
  if (projectionEl.value === 'xy') return [c[0], c[1]];
  if (projectionEl.value === 'xz') return [c[0], c[2]];
  if (projectionEl.value === 'yz') return [c[1], c[2]];
  return [
    c[0] * pca[0][0] + c[1] * pca[0][1] + c[2] * pca[0][2],
    c[0] * pca[1][0] + c[1] * pca[1][1] + c[2] * pca[1][2],
  ];
}

function rebuildPoints() {
  points = cameras.map(cameraPoint);
}

function fitView() {
  rebuildPoints();
  const rect = canvas.getBoundingClientRect();
  if (!points.length || rect.width <= 0 || rect.height <= 0) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }
  const pad = 44;
  const sx = (rect.width - pad * 2) / Math.max(1e-9, maxX - minX);
  const sy = (rect.height - pad * 2) / Math.max(1e-9, maxY - minY);
  view.scale = Math.max(1e-9, Math.min(sx, sy));
  view.tx = rect.width / 2 - ((minX + maxX) / 2) * view.scale;
  view.ty = rect.height / 2 + ((minY + maxY) / 2) * view.scale;
  draw();
}

function worldToScreen(p) {
  return [p[0] * view.scale + view.tx, -p[1] * view.scale + view.ty];
}

function screenToWorld(x, y) {
  return [(x - view.tx) / view.scale, -(y - view.ty) / view.scale];
}

function colorFor(cam, idx) {
  if (selectedSet.has(idx)) return '#c2410c';
  if (idx === focusIndex) return '#2563eb';
  const mode = colorModeEl.value;
  if (mode === 'selected') return '#667085';
  if (mode === 'name') {
    const h = hashString(cam.name) % 360;
    return `hsl(${h}, 64%, 42%)`;
  }
  const minZ = heightRange[0], maxZ = heightRange[1];
  const t = (cam.center[2] - minZ) / Math.max(1e-9, maxZ - minZ);
  const hue = 210 - 170 * t;
  return `hsl(${hue}, 68%, 40%)`;
}

function hashString(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function drawGrid(rect) {
  ctx.save();
  ctx.strokeStyle = '#dfe5ec';
  ctx.lineWidth = 1;
  const stepWorld = niceStep(80 / view.scale);
  const a = screenToWorld(0, rect.height);
  const b = screenToWorld(rect.width, 0);
  const minX = Math.floor(a[0] / stepWorld) * stepWorld;
  const maxX = Math.ceil(b[0] / stepWorld) * stepWorld;
  const minY = Math.floor(a[1] / stepWorld) * stepWorld;
  const maxY = Math.ceil(b[1] / stepWorld) * stepWorld;
  for (let x = minX; x <= maxX; x += stepWorld) {
    const s = worldToScreen([x, 0])[0];
    ctx.beginPath(); ctx.moveTo(s, 0); ctx.lineTo(s, rect.height); ctx.stroke();
  }
  for (let y = minY; y <= maxY; y += stepWorld) {
    const s = worldToScreen([0, y])[1];
    ctx.beginPath(); ctx.moveTo(0, s); ctx.lineTo(rect.width, s); ctx.stroke();
  }
  ctx.restore();
}

function niceStep(v) {
  const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
  const m = v / e;
  if (m < 2) return 2 * e;
  if (m < 5) return 5 * e;
  return 10 * e;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!cameras.length) return;
  drawGrid(rect);

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#0f766e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < selected.length; i++) {
    const s = worldToScreen(points[selected[i]]);
    if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
  }
  ctx.stroke();
  ctx.restore();

  const radius = Math.max(2, Math.min(6, 3.5 + Math.log10(view.scale + 1)));
  for (let i = 0; i < cameras.length; i++) {
    const s = worldToScreen(points[i]);
    if (s[0] < -20 || s[0] > rect.width + 20 || s[1] < -20 || s[1] > rect.height + 20) continue;
    ctx.beginPath();
    ctx.fillStyle = colorFor(cameras[i], i);
    ctx.arc(s[0], s[1], selectedSet.has(i) ? radius + 2 : radius, 0, Math.PI * 2);
    ctx.fill();
    if (i === hover || i === focusIndex) {
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  ctx.fillStyle = '#475467';
  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  ctx.fillText(`${projectionEl.value.toUpperCase()} | wheel zoom, drag pan, click camera to append`, 14, 22);
}

function screenDistance2(idx, x, y) {
  const s = worldToScreen(points[idx]);
  const dx = s[0] - x, dy = s[1] - y;
  return dx * dx + dy * dy;
}

function nearestCamera(x, y, current = -1) {
  const maxDist2 = 14 * 14;
  if (current >= 0 && screenDistance2(current, x, y) <= maxDist2) {
    return current;
  }
  let best = -1;
  let bestD = maxDist2;
  for (let i = 0; i < points.length; i++) {
    const d = screenDistance2(i, x, y);
    if (d < bestD || (Math.abs(d - bestD) < 1e-6 && best >= 0 && cameras[i].name < cameras[best].name)) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function appendCamera(idx) {
  if (idx < 0 || selectedSet.has(idx)) return;
  selected.push(idx);
  selectedSet.add(idx);
  focusIndex = idx;
  updateSelectedList();
  draw();
}

function updateSelectedList() {
  selectedCount.textContent = selected.length;
  const start = Math.max(0, selected.length - 300);
  selectedList.innerHTML = '';
  for (let order = start; order < selected.length; order++) {
    const idx = selected[order];
    const cam = cameras[idx];
    const row = document.createElement('div');
    row.className = 'item';
    row.innerHTML = `<div class="idx">${order}</div><div class="name" title="${cam.name}">${cam.name}</div><button class="mini">Focus</button>`;
    row.querySelector('button').onclick = () => focusCamera(idx);
    selectedList.appendChild(row);
  }
  selectedList.scrollTop = selectedList.scrollHeight;
}

function pointerPosition(ev) {
  const rect = canvas.getBoundingClientRect();
  return [ev.clientX - rect.left, ev.clientY - rect.top];
}

function focusCamera(idx) {
  if (idx < 0) return;
  focusIndex = idx;
  const s = worldToScreen(points[idx]);
  const rect = canvas.getBoundingClientRect();
  view.tx += rect.width / 2 - s[0];
  view.ty += rect.height / 2 - s[1];
  draw();
}

async function saveMapping() {
  const rows = selected.map((idx, order) => ({
    order,
    name: cameras[idx].name,
    center: cameras[idx].center,
  }));
  setStatus(`Saving mapping.txt with ${rows.length} selected cameras...`);
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ order: rows }),
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus(data.error || 'Save failed');
    return;
  }
  setStatus(`Saved subset mapping: ${data.count} rows to ${data.path}`);
}

async function loadCameras() {
  const res = await fetch('/api/cameras');
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Failed to load cameras');
  cameras = data.cameras;
  pca = data.pca;
  const zVals = cameras.map(c => c.center[2]);
  heightRange = zVals.length ? [Math.min(...zVals), Math.max(...zVals)] : [0, 1];
  datasetPath.textContent = data.dataset;
  cameraCount.textContent = cameras.length;
  const rangeText = data.range_label ? ` (${data.range_label})` : '';
  setStatus(`Loaded ${cameras.length} / ${data.total_cameras} cameras${rangeText}`);
  rebuildPoints();
  fitView();
}

canvas.addEventListener('pointerdown', (ev) => {
  ev.preventDefault();
  canvas.setPointerCapture(ev.pointerId);
  drag = { x: ev.clientX, y: ev.clientY, moved: false, pointerId: ev.pointerId };
});
canvas.addEventListener('pointermove', (ev) => {
  ev.preventDefault();
  const [x, y] = pointerPosition(ev);
  if (drag) {
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
    view.tx += dx; view.ty += dy;
    drag.x = ev.clientX; drag.y = ev.clientY;
    draw();
    return;
  }
  hover = nearestCamera(x, y, hover);
  if (hover >= 0) {
    const cam = cameras[hover];
    const order = selected.indexOf(hover);
    tooltip.style.display = 'block';
    tooltip.style.left = `${x + 14}px`;
    tooltip.style.top = `${y + 14}px`;
    tooltip.textContent = `name: ${cam.name}\ncenter: ${cam.center.map(v => v.toFixed(4)).join(', ')}${order >= 0 ? `\norder: ${order}` : ''}`;
  } else {
    tooltip.style.display = 'none';
  }
  draw();
});
canvas.addEventListener('pointerup', (ev) => {
  ev.preventDefault();
  if (!drag) return;
  const wasDrag = drag.moved;
  try { canvas.releasePointerCapture(drag.pointerId); } catch (err) {}
  drag = null;
  if (!wasDrag) {
    const [x, y] = pointerPosition(ev);
    appendCamera(nearestCamera(x, y));
  }
});
canvas.addEventListener('pointercancel', () => {
  drag = null;
});
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const before = screenToWorld(mx, my);
  const factor = Math.exp(-ev.deltaY * 0.001);
  view.scale *= factor;
  const after = worldToScreen(before);
  view.tx += mx - after[0];
  view.ty += my - after[1];
  draw();
}, { passive: false });

document.getElementById('fitBtn').onclick = fitView;
document.getElementById('undoBtn').onclick = () => {
  const idx = selected.pop();
  if (idx !== undefined) selectedSet.delete(idx);
  focusIndex = selected.length ? selected[selected.length - 1] : -1;
  updateSelectedList();
  draw();
};
document.getElementById('clearBtn').onclick = () => {
  if (!selected.length || confirm('Clear current manual order?')) {
    selected = [];
    selectedSet = new Set();
    focusIndex = -1;
    updateSelectedList();
    draw();
  }
};
document.getElementById('saveBtn').onclick = saveMapping;
projectionEl.onchange = () => { rebuildPoints(); fitView(); };
colorModeEl.onchange = draw;
document.getElementById('goBtn').onclick = () => {
  const q = searchEl.value.trim().toLowerCase();
  if (!q) return;
  let idx = cameras.findIndex(c => c.name.toLowerCase() === q);
  if (idx < 0) {
    idx = cameras.findIndex(c => c.name.toLowerCase().replace(/\.[^.]+$/, '') === q);
  }
  if (idx < 0) {
    idx = cameras.findIndex(c => c.name.toLowerCase().includes(q));
  }
  if (idx >= 0) {
    focusCamera(idx);
    setStatus(`Focused ${cameras[idx].name}`);
  } else {
    setStatus(`No image name matched: ${q}`);
  }
};
searchEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') document.getElementById('goBtn').click();
});
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(document.getElementById('stageWrap'));

resize();
loadCameras().catch(err => setStatus(err.message));
</script>
</body>
</html>
"""


def read_cameras_binary_fast(path):
    cameras = []
    with path.open("rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            data = f.read(64)
            if len(data) != 64:
                raise ValueError(f"Unexpected EOF while reading {path}")
            values = struct.unpack("<idddddddi", data)
            image_id = values[0]
            qvec = np.array(values[1:5], dtype=np.float64)
            tvec = np.array(values[5:8], dtype=np.float64)
            camera_id = values[8]
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"":
                    raise ValueError(f"Unexpected EOF while reading image name in {path}")
                if ch == b"\x00":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * num_points2d, os.SEEK_CUR)
            cameras.append(make_camera_record(image_id, camera_id, name, qvec, tvec))
    return cameras


def read_cameras_text_fast(path):
    cameras = []
    with path.open("r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])), dtype=np.float64)
            tvec = np.array(tuple(map(float, elems[5:8])), dtype=np.float64)
            camera_id = int(elems[8])
            name = elems[9]
            f.readline()
            cameras.append(make_camera_record(image_id, camera_id, name, qvec, tvec))
    return cameras


def make_camera_record(image_id, camera_id, name, qvec, tvec):
    center = -qvec2rotmat(qvec).T @ tvec
    return {
        "image_id": int(image_id),
        "camera_id": int(camera_id),
        "name": name,
        "center": [float(center[0]), float(center[1]), float(center[2])],
    }


def find_sparse_dir(dataset):
    dataset = dataset.resolve()
    if (dataset / "images.bin").is_file() or (dataset / "images.txt").is_file():
        return dataset
    sparse = dataset / "sparse" / "0"
    if sparse.is_dir():
        return sparse
    raise FileNotFoundError(f"Cannot find COLMAP images.bin/images.txt under {dataset} or {sparse}")


def load_cameras(dataset):
    sparse = find_sparse_dir(dataset)
    bin_path = sparse / "images.bin"
    txt_path = sparse / "images.txt"
    if bin_path.is_file():
        cameras = read_cameras_binary_fast(bin_path)
    elif txt_path.is_file():
        cameras = read_cameras_text_fast(txt_path)
    else:
        raise FileNotFoundError(f"Cannot find images.bin or images.txt in {sparse}")
    cameras.sort(key=lambda c: c["name"])
    return sparse, cameras


def image_name_index(name):
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return None


def filter_cameras_by_name_range(cameras, start, end):
    if start is None and end is None:
        return cameras
    filtered = []
    for camera in cameras:
        index = image_name_index(camera["name"])
        if index is None:
            continue
        if start is not None and index < start:
            continue
        if end is not None and index > end:
            continue
        filtered.append(camera)
    return filtered


def range_label(start, end):
    if start is None and end is None:
        return ""
    if start is None:
        return f"name <= {end:04d}"
    if end is None:
        return f"name >= {start:04d}"
    return f"{start:04d} <= name <= {end:04d}"


def compute_pca_axes(cameras):
    xyz = np.array([c["center"] for c in cameras], dtype=np.float64)
    if len(xyz) < 2:
        return [[1, 0, 0], [0, 1, 0]]
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(xyz, full_matrices=False)
    axes = vh[:2]
    return [[float(v) for v in row] for row in axes]


def write_mapping(path, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# order image_name center_x center_y center_z\n")
        for item in order:
            center = item["center"]
            f.write(
                f"{item['order']} {item['name']} "
                f"{center[0]:.12g} {center[1]:.12g} {center[2]:.12g}\n"
            )


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "ColmapOrderGUI/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/cameras":
            self.send_json(
                {
                    "dataset": str(self.server.dataset),
                    "sparse": str(self.server.sparse),
                    "mapping_path": str(self.server.mapping_path),
                    "pca": self.server.pca_axes,
                    "total_cameras": self.server.total_cameras,
                    "range_label": self.server.range_label,
                    "cameras": self.server.cameras,
                }
            )
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            order = payload.get("order", [])
            if not isinstance(order, list):
                raise ValueError("order must be a list")
            write_mapping(self.server.mapping_path, order)
            self.send_json({"path": str(self.server.mapping_path), "count": len(order)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data).encode("utf-8"), "application/json", status)

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


class GuiServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler, dataset, mapping_path, start, end):
        self.dataset = dataset.resolve()
        self.sparse, cameras = load_cameras(self.dataset)
        self.total_cameras = len(cameras)
        self.cameras = filter_cameras_by_name_range(cameras, start, end)
        self.range_label = range_label(start, end)
        self.pca_axes = compute_pca_axes(self.cameras)
        self.mapping_path = mapping_path.resolve()
        super().__init__(server_address, handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a browser GUI to inspect COLMAP cameras and save manual order mapping.")
    parser.add_argument("--dataset", "-d", required=True, help="COLMAP root containing sparse/0, or the sparse model folder itself.")
    parser.add_argument("--mapping", "-m", default=None, help="Output mapping.txt path. Defaults to <dataset>/mapping.txt.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Use 0.0.0.0 for remote access if needed.")
    parser.add_argument("--port", type=int, default=7860, help="Port for VS Code forwarding.")
    parser.add_argument("--start", type=int, default=None, help="Only show images whose numeric filename stem is >= start.")
    parser.add_argument("--end", type=int, default=None, help="Only show images whose numeric filename stem is <= end.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start is not None and args.end is not None and args.start > args.end:
        raise ValueError("--start must be <= --end")
    dataset = Path(args.dataset)
    mapping_path = Path(args.mapping) if args.mapping else dataset / "mapping.txt"
    server = GuiServer((args.host, args.port), GuiHandler, dataset, mapping_path, args.start, args.end)
    range_text = f" [{server.range_label}]" if server.range_label else ""
    print(f"Loaded {len(server.cameras)} / {server.total_cameras} cameras from {server.sparse}{range_text}", flush=True)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    print(f"Mapping will be saved to {server.mapping_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
