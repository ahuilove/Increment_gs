# Ahuilove的离线增量式重建探索
道阻且长 行则将至

# Stage1: Graph Chunking And Feed-Forward SfM Experiments

本目录用于把一个已有的 COLMAP 数据集切成带 overlap 的 chunk，并在单个 chunk 上测试
MASt3R-SfM、VGGT-X、VGGT-Omega 等重建路线，最后尽量导出成 3DGS 可读取的 COLMAP 格式。

下面的命令都假设在项目根目录运行：

```bash
cd /home/zhanqh/WorkSpace/3d_related/city_generation/city_reconstruction/Increment_gs
```

示例原始数据集路径：

```bash
DATASET=/home/zhanqh/dataset/city/Matrixcity/small_city/aerial/train/block_all
```

原始数据集需要是标准 COLMAP 结构：

```text
block_all/
  images/
  sparse/0/
    cameras.bin 或 cameras.txt
    images.bin 或 images.txt
    points3D.bin 或 points3D.txt
```

`stage1/chunk/`、`stage1/oracle_camera_graph.json`、`stage1/checkpoint/` 已加入 `.gitignore`，
适合放中间结果和大模型权重。

## 1. 生成 Oracle Camera Graph

脚本：`stage1/camera_graph.py`

节点是图片。脚本读取 COLMAP `sparse/0` 中每张图片的相机位姿，计算相机中心距离和朝向夹角，
并为每张图片保留 topK 个空间/视角最相关的邻居。

基础打分：

```text
score(i, j) = exp(-dist(i,j) / sigma_d) * max(0, dot(forward_i, forward_j))
```

常用命令：

```bash
python stage1/camera_graph.py "$DATASET" \
  --output stage1/oracle_camera_graph.json \
  --topk 20 \
  --max_dist 100 \
  --max_angle_deg 60 \
  --sigma_d 10
```

主要参数：

- `--topk`: 每张图最多保留多少条邻居边。
- `--max_dist`: 相机中心最大距离，超过则不连边；不传表示不限制。
- `--max_angle_deg`: forward 方向最大夹角，默认 90 度。
- `--sigma_d`: 距离衰减尺度，越大表示远距离惩罚越弱。

输出 JSON 形如：

```json
{
  "image_0001.png": [
    {"image": "image_0002.png", "score": 0.93, "dist": 3.1, "angle_deg": 12.0}
  ]
}
```

## 2. 根据 Graph 生成 Chunk

脚本：`stage1/graph_driven_chunk_generation.py`

算法思想：

- 从未覆盖图片中选择一个 seed。
- 沿 graph 的高分边不断扩展，直到达到 `chunk_size`。
- 当前 chunk 选中的主图片是 `core`。
- 已被其他 chunk 覆盖、但被当前 chunk 引用的图片作为 `overlap`。
- 每个 chunk 会导出一个合法的 COLMAP 子数据集。

常用命令：

```bash
python stage1/graph_driven_chunk_generation.py \
  --graph stage1/oracle_camera_graph.json \
  --input "$DATASET" \
  --output stage1/chunk \
  --chunk_size 80 \
  --min_overlap 10 \
  --overlap_ratio 0.2 \
  --copy-mode symlink \
  --overwrite
```

输出结构：

```text
stage1/chunk/
  chunk_0000/
    images/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
    mapping.txt
    chunk_metadata.json
  chunk_0001/
    ...
```

`mapping.txt` 记录当前 chunk 内图片和原始数据集图片的对应关系，以及该图片在 chunk 中是
`core` 还是 `overlap`。这对后续把 chunk 结果映射回全局数据集很重要。

## 3. 使用 MASt3R-SfM 并转成 COLMAP

相关脚本：

- `stage1/mast3r_sfm_reconstruct.py`: 只运行 MASt3R-SfM 的 `sparse_global_alignment`，保存 `scene.pt`。
- `stage1/mast3r_sfm_to_colmap.py`: 将 `scene.pt` 导出成 COLMAP-like `sparse/0`。
- `stage1/visualize_mast3r_scene.py`: 将 `scene.pt` 导出 PLY，方便本地可视化。
- `stage1/eval_metric.py`: 将预测 COLMAP 和真值 COLMAP 做相机位姿误差评估。

### 3.1 准备 MASt3R checkpoint

建议放到：

```text
stage1/checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

如果要手动下载，需要保证文件名和传入的 `--weights` 一致。

### 3.2 跑 MASt3R-SfM

```bash
python stage1/mast3r_sfm_reconstruct.py \
  --images stage1/chunk/chunk_0000/images \
  --output stage1/chunk/chunk_0000/mast3r_sfm \
  --weights stage1/checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  --device cuda \
  --image_size 512 \
  --scene_graph complete
```

输出：

```text
stage1/chunk/chunk_0000/mast3r_sfm/
  scene.pt
  image_list.txt
  config.json
  cache/
```

如果想先看 MASt3R-SfM 原始结果：

```bash
python stage1/visualize_mast3r_scene.py \
  --scene stage1/chunk/chunk_0000/mast3r_sfm/scene.pt \
  --output stage1/chunk/chunk_0000/mast3r_sfm/vis \
  --max_points_per_image 5000
```

`--max_points_per_image` 表示每张图最多导出多少个 MASt3R sparse point 到 PLY。值越大点云越密，
文件也越大；小一点更方便快速检查相机和点云是否明显错误。

### 3.3 MASt3R-SfM 转 COLMAP

```bash
python stage1/mast3r_sfm_to_colmap.py \
  --scene stage1/chunk/chunk_0000/mast3r_sfm/scene.pt \
  --output stage1/chunk/chunk_0000/mast3r_colmap \
  --images stage1/chunk/chunk_0000/images \
  --copy-mode symlink \
  --output-format .bin \
  --overwrite
```

注意：当前第一版导出使用 `per_view_points`。也就是说，每张图的点是从该视角独立导出的，
每个 3D 点只有单视角 observation，不是 COLMAP 那种多视角 track-aware triangulation。
它的目标是先生成合法的 COLMAP 结构，方便接 3DGS/评估流程。

### 3.4 计算和真值 COLMAP 的误差

如果 chunk 自带真值 `sparse/0`，可以直接作为 GT：

```bash
python stage1/eval_metric.py \
  --pred-sparse-dir stage1/chunk/chunk_0000/mast3r_colmap/sparse/0 \
  --sparse-dir stage1/chunk/chunk_0000/sparse/0 \
  --thresholds 3 5 10 30
```

## 4. 使用 VGGT-X 并导出 COLMAP

相关脚本：

- `stage1/vggt_x_colmap_wrapper.py`

这个 wrapper 不修改 `submodules/vggt-x`，而是在主项目中：

- 将 `submodules/vggt-x` 加入 `sys.path`。
- patch `demo_colmap.run_VGGT`，从本地 `stage1/checkpoint/vggt_1b_model.pt` 读取权重。
- 默认关闭 `torch.compile`，避免部分 PyTorch/Jinja2/Inductor 环境在 import 阶段报错。

准备权重：

```text
stage1/checkpoint/vggt_1b_model.pt
```

运行：

```bash
python stage1/vggt_x_colmap_wrapper.py \
  --scene_dir stage1/chunk/chunk_0000 \
  --checkpoint stage1/checkpoint/vggt_1b_model.pt \
  --use_ga \
  --shared_camera \
  --chunk_size 128 \
  --max_points_for_colmap 500000
```

VGGT-X 的输出目录由 submodule 里的 `demo_colmap.py` 决定：

```text
stage1/chunk_vggt_x/chunk_0000/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  sparse/points.ply
```

如果传入 `--post_fix _vggt_x`，并且输入是 `stage1/chunk/chunk_0000`，输出就是
`stage1/chunk_vggt_x/chunk_0000`。

评估：

```bash
python stage1/eval_metric.py \
  --pred-sparse-dir stage1/chunk_vggt_x/chunk_0000/sparse/0 \
  --sparse-dir stage1/chunk/chunk_0000/sparse/0 \
  --thresholds 3 5 10 30
```

常见环境问题：

- 如果 `pycolmap` 报 `module 'pycolmap' has no attribute 'Reconstruction'`，通常是装到了错误的
  `pycolmap 0.0.1`。需要安装真正的 COLMAP Python binding，例如 `pycolmap==3.10.0`。
- 如果 `torch.compile` 触发 Jinja2/Inductor 报错，默认 wrapper 已禁用；只有确实需要时再传
  `--enable_torch_compile`。
- `chunk_size` 是 VGGT-X 一次处理的帧块大小。图片数量少于 `chunk_size` 也可以运行。

## 5. 使用 VGGT-Omega 并导出 COLMAP

相关脚本：

- `stage1/demo_vggt_omega.py`: 从 `submodules/vggt-omega` 迁移来的命令行 demo，负责输出 `predictions.npz`。
- `stage1/vggtomega2colmap.py`: 将 `predictions.npz` 转为 COLMAP-like 数据集，并可额外输出 3DGS depth supervision。
- `stage1/eval_metric.py`: 支持直接评估 `predictions.npz` 或转换后的 COLMAP。

### 5.1 跑 VGGT-Omega

准备 VGGT-Omega checkpoint，例如：

```text
stage1/checkpoint/vggt_omega.pt
```

运行：

```bash
python stage1/demo_vggt_omega.py \
  --images stage1/chunk/chunk_0000/images \
  --checkpoint stage1/checkpoint/vggt_omega_1b_512.pt \
  --output-dir stage1/chunk/chunk_0000/vggtomega \
  --image-resolution 512 \
  --preprocess-mode balanced \
  --overwrite
```

关键输出：

```text
stage1/chunk/chunk_0000/vggtomega/
  predictions.npz
  images/
  ...
```

`predictions.npz` 中通常包含 VGGT-Omega 预测的 `extrinsic`、`intrinsic`、`depth`、
`depth_conf`、`world_points_from_depth` 等结果。

### 5.2 VGGT-Omega 转 COLMAP

```bash
python stage1/vggtomega2colmap.py \
  --predictions stage1/chunk/chunk_0000/vggtomega/predictions.npz \
  --output stage1/chunk/chunk_0000/vggtomega_colmap \
  --max-points 500000 \
  --copy-mode symlink \
  --overwrite
```

输出结构：

```text
stage1/chunk/chunk_0000/vggtomega_colmap/
  images/
  depth/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
    depth_params.json
  conversion_summary.json
```

当前转换逻辑：

- 根据 VGGT-Omega 的预处理方式，把预测的内参从处理后图像坐标映射回原始图片坐标。
- 使用 `world_points_from_depth` 或 `depth + extrinsic + intrinsic` 生成点云。
- 按 `depth_conf` 过滤并最多导出 `--max-points` 个 3D 点。
- 每个 3D 点目前也是单视角 observation，不是 COLMAP track-aware triangulation。

`--copy-mode` 控制输出目录的 `images/` 如何引用原图：

- `symlink`: 默认，省空间。
- `hardlink`: 同磁盘时省空间。
- `copy`: 复制图片，最独立但占空间。
- `none`: 不创建 `images/`，适合只想导出 sparse。

### 5.3 深度图格式和 3DGS 训练

`vggtomega2colmap.py` 默认会输出 `depth/` 和 `sparse/0/depth_params.json`。

这里的 depth 是 3DGS `train.py --depths` 可读取的 inverse-depth PNG：

- `depth/<image_basename>.png` 是 `uint16`。
- `train.py` 会先读成 `png / 2**16`。
- `depth_params.json` 中的 `scale` 和 `offset` 用于恢复 inverse depth 的尺度。
- 默认 `--depth-normalization global_scale` 会避免 inverse depth 直接写 uint16 时饱和。

用 VGGT-Omega COLMAP 输出训练 3DGS：

```bash
python train.py \
  -s stage1/chunk/chunk_0000/vggtomega_colmap \
  -m output/chunk_0000_vggtomega_3dgs_depth \
  --depths depth \
  --disable_viewer
```

如果只想先短跑测试：

```bash
python train.py \
  -s stage1/chunk/chunk_0000/vggtomega_colmap \
  -m output/chunk_0000_vggtomega_3dgs_depth_test \
  --depths depth \
  --iterations 3000 \
  --test_iterations 1000 3000 \
  --save_iterations 1000 3000 \
  --disable_viewer
```

### 5.4 评估 VGGT-Omega

直接评估 `predictions.npz`：

```bash
python stage1/eval_metric.py \
  --predictions stage1/chunk/chunk_0000/vggtomega/predictions.npz \
  --images-dir stage1/chunk/chunk_0000/vggtomega/images \
  --sparse-dir stage1/chunk/chunk_0000/sparse/0 \
  --thresholds 3 5 10 30
```

评估转换后的 COLMAP：

```bash
python stage1/eval_metric.py \
  --pred-sparse-dir stage1/chunk/chunk_0000/vggtomega_colmap/sparse/0 \
  --sparse-dir stage1/chunk/chunk_0000/sparse/0 \
  --thresholds 3 5 10 30
```

## 推荐的新数据集完整流程

给一个新的 COLMAP 数据集，推荐先按下面顺序跑通一个 chunk：

```bash
DATASET=/path/to/new_colmap_dataset

python stage1/camera_graph.py "$DATASET" \
  --output stage1/oracle_camera_graph.json \
  --topk 20 \
  --max_angle_deg 60 \
  --sigma_d 10

python stage1/graph_driven_chunk_generation.py \
  --graph stage1/oracle_camera_graph.json \
  --input "$DATASET" \
  --output stage1/chunk \
  --chunk_size 80 \
  --min_overlap 10 \
  --overlap_ratio 0.2 \
  --copy-mode symlink \
  --overwrite
```

然后在 `stage1/chunk/chunk_0000` 上分别测试：

- MASt3R-SfM: `mast3r_sfm_reconstruct.py` -> `mast3r_sfm_to_colmap.py` -> `eval_metric.py`
- VGGT-X: `vggt_x_colmap_wrapper.py` -> `eval_metric.py`
- VGGT-Omega: `demo_vggt_omega.py` -> `vggtomega2colmap.py` -> `eval_metric.py` 或 `train.py`

先确认单个 chunk 的相机位姿、点云和 3DGS 训练能跑通，再扩大到更多 chunk。
