# Ahuilove的离线增量式重建探索
道阻且长 行则将至

## 文档入口

- `stage1/README.md`: 记录 chunk 生成、MASt3R-SfM、VGGT-X、VGGT-Omega 到 COLMAP/3DGS 的流程。
- `eval/`: 记录统一轨迹评价工具，主要用于比较预测 COLMAP 轨迹和真值 COLMAP 轨迹。

下面的命令默认在项目根目录运行：

```bash
cd /home/zhanqh/WorkSpace/3d_related/city_generation/city_reconstruction/Increment_gs
```

## Eval: 相机轨迹评价

`eval` 目录用于把不同方法输出的相机轨迹统一到 COLMAP sparse 格式，然后计算轨迹误差并画轨迹图。

当前包含：

- `eval/traj_eval.py`: 输入预测 COLMAP sparse 和真值 COLMAP sparse，输出 `RPE_t`、`RPE_r`、`ATE` 和轨迹图。
- `eval/longsplat_json2colmap.py`: 将 LongSplat 的 `cameras_all_train.json` 或 `cameras_all_test.json` 转成 COLMAP sparse，供 `traj_eval.py` 使用。

### 1. 直接评价两个 COLMAP sparse

如果预测结果已经是 COLMAP 格式，例如：

```text
pred/sparse/0/
  cameras.bin 或 cameras.txt
  images.bin 或 images.txt
  points3D.bin 或 points3D.txt

gt/sparse/0/
  cameras.bin 或 cameras.txt
  images.bin 或 images.txt
  points3D.bin 或 points3D.txt
```

运行：

```bash
python eval/traj_eval.py \
  --pred-sparse-dir /path/to/pred/sparse/0 \
  --gt-sparse-dir /path/to/gt/sparse/0 \
  --output eval/output_traj
```

也可以传数据集根目录或 `sparse` 目录，脚本会自动寻找 `sparse/0`：

```bash
python eval/traj_eval.py \
  --pred-sparse-dir /path/to/pred_dataset \
  --gt-sparse-dir /path/to/gt_dataset \
  --output eval/output_traj
```

输出：

```text
eval/output_traj/
  metrics.json
  metrics.txt
  matched_poses.csv
  pose_vis.png
  pose_vis.names.txt
```

其中：

- `metrics.txt`: 简洁指标文本。
- `metrics.json`: 完整指标和对齐信息。
- `matched_poses.csv`: 每张匹配图像的 GT、预测、对齐后预测相机中心。
- `pose_vis.png`: 使用 evo 风格画出的预测轨迹和 GT 轨迹。
- `pose_vis.names.txt`: 参与评价的图像名顺序。

### 2. 指标含义

`traj_eval.py` 默认会先把预测轨迹通过 Sim3 对齐到 GT：

```text
--alignment sim3
```

这适合 MASt3R、VGGT、LongSplat 这类可能存在尺度不确定性的结果。

输出指标：

- `RPE_t`: 相邻帧 Relative Pose Error 的平移部分，原始值。
- `RPE_t_x100`: `RPE_t * 100`，用于和 InstantSplat/LongSplat 的打印习惯保持一致。
- `RPE_r_deg`: 相邻帧 Relative Pose Error 的旋转部分，单位是度。
- `ATE`: Absolute Trajectory Error，对齐后每帧相机中心误差的 RMSE。

例如：

```text
RPE_t: 1.7720281 (x100)
RPE_r: 0.5456420 deg
ATE  : 0.1347615
```

等价于：

```text
RPE_t_raw = 0.017720281
RPE_t_x100 = 1.7720281
RPE_r_deg = 0.5456420
ATE = 0.1347615
```

### 3. 常用参数

```bash
python eval/traj_eval.py \
  --pred-sparse-dir /path/to/pred/sparse/0 \
  --gt-sparse-dir /path/to/gt/sparse/0 \
  --output eval/output_traj \
  --alignment sim3 \
  --match-by basename
```

参数说明：

- `--alignment`: 可选 `sim3`、`se3`、`none`。默认 `sim3`。
- `--match-by basename`: 默认只用图片 basename 匹配，忽略 COLMAP image name 中的目录前缀。
- `--match-by name`: 使用完整 COLMAP image name 匹配。
- `--no-plot`: 不输出轨迹图。
- `--plot-video-frames`: 额外输出逐帧累积轨迹图到 `output/pose_vid/`，类似 InstantSplat 的 `vid=True`。
- `--no-rpe-percent`: 只打印原始 `RPE_t`，不打印 `x100` 版本。

### 4. 评价 LongSplat 输出

LongSplat 训练结束后会输出：

```text
<longsplat_model_path>/
  cameras_all_train.json
  cameras_all_test.json
```

如果 `train.py` 不加 `--eval`，所有图片都会进入 train set，此时主要评价：

```text
cameras_all_train.json
```

如果加了 `--eval`，LongSplat 会按固定间隔划分 train/test：

- `custom/free`: 默认 `idx % 8 == 0` 为 test。
- `hike`: 默认 `idx % 10 == 0` 为 test。
- `tanks`: 通常每 8 张取 1 张 test，`Family` 场景为每 2 张取 1 张。

注意：test pose 需要在跑完 `render.py` 后才会被 `vis_loc` 更新到 `cameras_all_test.json`。只跑 `train.py` 时，可靠的是 `cameras_all_train.json`。

#### 4.1 LongSplat JSON 转 COLMAP

例如评价 grass 的 LongSplat train 轨迹：

```bash
python eval/longsplat_json2colmap.py \
  --input baseline/LongSplat/outputs/free/grass/cameras_all_train.json \
  --output eval/longsplat_grass_train_colmap \
  --ext .txt \
  --image-ext .JPG \
  --overwrite
```

输出：

```text
eval/longsplat_grass_train_colmap/
  sparse/0/
    cameras.txt
    images.txt
    points3D.txt
  longsplat_json2colmap_summary.json
```

`--image-ext` 很重要。LongSplat 的 JSON 中 `image_name` 通常没有后缀，例如 `DSC07685`；但 GT COLMAP 里可能是 `DSC07685.JPG`。如果后缀不一致，`traj_eval.py` 会匹配不到图像。

如果传入图片目录，脚本也可以自动从真实图片推断后缀：

```bash
python eval/longsplat_json2colmap.py \
  --input baseline/LongSplat/outputs/free/grass/cameras_all_train.json \
  --output eval/longsplat_grass_train_colmap \
  --copy-images-from /home/zhanqh/dataset/city/free_dataset/grass/images \
  --copy-mode symlink \
  --ext .txt \
  --overwrite
```

#### 4.2 评价转换后的 LongSplat 轨迹

```bash
python eval/traj_eval.py \
  --pred-sparse-dir eval/longsplat_grass_train_colmap/sparse/0 \
  --gt-sparse-dir /home/zhanqh/dataset/city/free_dataset/grass/sparse/0 \
  --output eval/longsplat_grass_train_eval
```

如果评价 test 轨迹，先确保已经跑过：

```bash
cd baseline/LongSplat
python render.py -m outputs/free/grass
cd ../..
```

然后转换并评价：

```bash
python eval/longsplat_json2colmap.py \
  --input baseline/LongSplat/outputs/free/grass/cameras_all_test.json \
  --output eval/longsplat_grass_test_colmap \
  --ext .txt \
  --image-ext .JPG \
  --overwrite

python eval/traj_eval.py \
  --pred-sparse-dir eval/longsplat_grass_test_colmap/sparse/0 \
  --gt-sparse-dir /home/zhanqh/dataset/city/free_dataset/grass/sparse/0 \
  --output eval/longsplat_grass_test_eval
```

### 5. 注意事项

- 预测和真值通过图像名匹配，不通过 COLMAP image id 匹配。
- COLMAP 存的是 world-to-camera，`traj_eval.py` 内部会转成 camera-to-world 后评价。
- 如果匹配数量为 0，优先检查图片后缀是否一致，例如 `.png`、`.jpg`、`.JPG`。
- 如果 RPE 和 LongSplat 自带 `pose_eval.txt` 不一致，优先检查评价的是 train 还是 test、图像顺序是否一致、是否都用了 Sim3 对齐。
- LongSplat 自带 `pose_eval.txt` 中：
  - `RPE_trans` 对应这里的 `RPE_t_x100`。
  - `RPE_rot` 对应这里的 `RPE_r_deg`。
  - `ATE` 对应这里的 `ATE`。
