# Kaggle 对接参考手册

BirdCLEF+ 2026 全流程中与 Kaggle 平台相关的知识、坑、流程的速查。本 repo 所有和 Kaggle 对接的经验都写在这里，按"我要做 X 时看什么"组织，便于 Cmd+F 查询。

## 目录

- [比赛信息](#比赛信息)
- [Kaggle 平台约束](#kaggle-平台约束)
- [数据集上传 (kaggle datasets)](#数据集上传-kaggle-datasets)
- [Kaggle GPU 训练流程](#kaggle-gpu-训练流程)
- [Code Competition 提交流程](#code-competition-提交流程)
- [提交 Notebook 的运行时装环境](#提交-notebook-的运行时装环境)
- [踩过的坑与修复清单](#踩过的坑与修复清单)
- [调试常用命令](#调试常用命令)

---

## 比赛信息

### 基本参数

| 项 | 值 |
|---|---|
| 比赛链接 | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) |
| 场景 | 南美 Pantanal 巴西湿地声学识别 |
| 类别数 | **234 个 primary_labels**（鸟类 + Amphibia + Insecta） |
| 采样率 | 32 kHz |
| 音频格式 | OGG |
| 窗口 | 5s × 12 = 60s per file |
| 指标 | Macro ROC-AUC（忽略没有正样本的类） |
| 比赛类型 | **Code Competition**（必须提交 notebook，不能传 CSV） |
| 推理硬件 | **CPU only**（GPU notebook 可提交但只有 1 分钟 runtime，等于没用） |
| 时间上限 | **CPU Notebook ≤ 90 分钟 run-time**（硬上限，超即 Notebook Timeout，不会保存中间结果） |
| Internet | 提交运行时**关闭** |
| 额外数据 | 允许用公开外部数据与预训练模型（Perch v2 等） |
| 提交文件 | 必须名为 `submission.csv`，输出到 `/kaggle/working/submission.csv` |

官方原文（[比赛 rules 页面](https://www.kaggle.com/competitions/birdclef-2026)）：

> - CPU Notebook **<= 90 minutes** run-time
> - GPU Notebook submissions are disabled. You can technically submit but will only have **1 minute** of runtime.
> - Internet access disabled
> - Freely & publicly available external data is allowed, including pre-trained models
> - Submission file must be named `submission.csv`

### 数据结构

```
data/birdclef-2026/
├── taxonomy.csv                     # 类别元信息 (primary_label, scientific, common, class)
├── sample_submission.csv            # 提交格式参考
├── train.csv                        # train_audio 元信息
├── train_soundscapes_labels.csv     # 708 条 labeled rows（5s 窗口级别）
├── recording_location.txt           # 站点经纬度
├── train_audio/                     # XC 录音（类别平衡差）
├── train_soundscapes/               # 软标签 soundscape + 708 hard labels
└── test_soundscapes/                # Public sample 不能看内容
```

### 关键观察

- **708 hard labels 分属仅 8 个 site**；训练集 soundscapes 覆盖 **~23 个 site**；测试集可能包含全部 23 个 site。→ 这是 site domain shift 的根源。
- **primary_labels** 不是 Perch 输出的顺序，需要 taxonomy 映射 + Amphibia/Insecta proxy 合并。
- **`train.csv` 的 primary_label** 是弱标签（文件级），不等同于窗口级标签。
- **test_soundscapes 文件数**：根据 v4.0 的 Perch-only 提交（~9.6s/file）刚好卡满 90min 逆推，估算 **~560 files**，约 6,700 个 5s 窗口。

### 评分

- Public LB 用 ~30% 测试集，Private LB 用剩余 70%
- 提交即评分，当天不限次数
- `submission.csv` 格式：`row_id,<primary_label_0>,<primary_label_1>,...` 每 5s 窗口一行

---

## Kaggle 平台约束

### 时间

**⚠️ 两个时限要区分**：

| 场景 | 硬上限 | 说明 |
|---|---|---|
| **Code Competition 提交**（本比赛 CPU）| **90 min** | 比赛规则强制，超即 Notebook Timeout 丢分 |
| Interactive notebook (普通使用) | 12h idle | 不活动 12h 被 kill |
| Save Version (非提交) | 12h | 普通开发测试 |
| GPU (T4/P100) 训练 | 12h per session | 本 repo 训 3 seed CNN 约 3h，充裕 |
| Session restart | 立即 | 状态丢失 |

**提交时限 ≠ 平台 session 时限**。提交的 90min 是比赛层规则，不是 Kaggle 平台层的。同一份 notebook：
- 点 **Save Version** 自由跑（Kaggle 给 12h）
- 点 **Submit to competition** 会严格卡 90min，超就丢

所以本地调试 Save Version 能跑 2 小时不 timeout，但一提交就挂，这不是 bug，这是规则。

### 算力配额

| 资源 | 周配额 |
|---|---|
| GPU (T4 / P100) | 30h/week |
| TPU VM | 20h/week（我们没用）|
| CPU | 无限 |

⚠️ 配额周一 UTC 0 点重置，不是滚动窗口。做 CNN 训练前确认当前剩余时间。

### 运行环境

- Python 版本：3.12（2026 起）
- 默认 conda：PyTorch 2.4、TensorFlow 2.15
- **没有 `onnxruntime`、`onnxscript`、TF 2.20**（本比赛都需要）
- GPU driver：CUDA 12.x
- `/kaggle/working/` 是持久化工作区（保存到 Version 里），`/tmp/` 临时
- `/kaggle/input/` 所有挂载数据集的根

### Internet 规则

- **Notebook 交互调试时**：Internet 可开（Settings → Internet on）
- **Save Version（含 Code Competition 提交）**：Internet **强制关闭**
- 这意味着 `pip install` 必须从**离线 wheel 文件**装

---

## 数据集上传 (kaggle datasets)

### 基本命令

```bash
# 首次创建
kaggle datasets create -p <dir> --dir-mode zip

# 更新版本
kaggle datasets version -p <dir> --dir-mode zip -m "description"

# 查看状态
kaggle datasets status -d <owner>/<slug>
kaggle datasets list -s birdclef
```

### `dataset-metadata.json` 模板

```json
{
  "title": "birdclef-2026",
  "id": "<your_username>/birdclef-2026",
  "licenses": [{"name": "CC0-1.0"}]
}
```

放在要上传的目录根下。`id` 的 slug 一旦 create 就不能改（只能 delete + 重建）。

### `--dir-mode` 详解（重要）

默认情况下 `kaggle datasets` 命令**跳过子目录**里的所有文件！

| Mode | 行为 |
|---|---|
| `skip` (默认) | 子目录整个丢掉，只上传根目录文件 |
| `zip` | 每个子目录打包成 `<name>.zip` 上传，Kaggle 自动解压为 `<name>/*` |
| `tar` | 同上但用 tar |

**本 repo 的 `artifacts/submission_bundle/wheels/` 必须用 `--dir-mode zip`**，否则 `onnxruntime` 离线安装会失败，CNN 直接被 disable。

Kaggle 自动解压后的挂载路径：`/kaggle/input/<owner>/<slug>/wheels/*.whl`（跟本地目录结构一致）。

### 上传完后路径

Kaggle notebook 里挂载一个 dataset，可能出现在以下路径之一，要全部 probe：

```python
_candidates = [
    Path(f'/kaggle/input/datasets/<owner>/<slug>'),  # notebook 挂载数据集
    Path(f'/kaggle/input/<slug>'),                    # 直接用 slug 挂载
]
```

### 版本更新陷阱

- **每个版本是完整快照**，不是增量。如果 v2 只传了 1 个文件，v1 里其他文件在 v2 **消失**。
- 版本号递增但**不显式可选**：`kaggle datasets version` 总是创建下一个版本。
- **notebook 重连延迟**：刚 push 完 dataset v5，在 Kaggle 网页 notebook 里可能仍看到 v4 几分钟。Force refresh 或重开 notebook。

### 我们的 3 个 datasets

| slug | 内容 | 关键文件 |
|---|---|---|
| `<you>/birdclef-2026` | 提交主 bundle + CNN ONNX + wheels | `submission_bundle.pkl`、`mel_cnn_seed*_fp16.onnx`、`wheels.zip` |
| `<you>/birdclef-2026-distill` | GPU 训练输入 | `teacher_cache_distill.pkl`（uint8 量化 teacher probs） |
| `<you>/birdclef-2026-cnn` | (可选) CNN ONNX 独立 dataset | 同上 onnx 文件 |

---

## Kaggle GPU 训练流程

### 1. 准备 distill cache（本地）

```bash
python scripts/07_prepare_cnn_distill.py
# 产出 artifacts/cnn_distill/teacher_cache_distill.pkl (~34 MB)
#       + dataset-metadata.json
```

`teacher_cache_distill.pkl` 字段：
- `meta_row_id`：窗口 ID（`stem_windowIdx`）
- `full_cache_probs`：uint8 (N, 234)，teacher 的 sigmoid 输出 ×255
- `primary_labels`：234 个
- `labeled_cache_idx`：708 条 labeled rows 在 full cache 里的 index
- `Y_full_truth`：(708, 234) 0/1 硬标签

uint8 量化：把 fp32 probs * 255 round 成 uint8，节省 4x 空间。解压时 `probs / 255.0` 即可。

### 2. 上传到 Kaggle

```bash
kaggle datasets create -p artifacts/cnn_distill --dir-mode zip
# 或 update: kaggle datasets version -p artifacts/cnn_distill --dir-mode zip -m "v2"
```

### 3. 在 Kaggle 创建训练 notebook

1. 上传 `kaggle_gpu/train_mel_cnn.ipynb` 到 Kaggle
2. **Data 挂载**：
   - `<you>/birdclef-2026-distill`
   - `birdclef-2026` competition（需要 `train_soundscapes/*.ogg`）
3. **Settings → Runtime**：
   - Accelerator：**T4 × 1**（15 GB VRAM；比 P100 更稳定）
   - Internet：On（训练时可开，因为要下 PyTorch / torchaudio 依赖）
   - Persistence：Variables & Files
4. **Save & Run All**

### 4. 训练参数（已在 notebook 里调好）

```python
# CFG
seeds         = [20260101, 20260215, 20260322]
epochs        = 18    # 每 seed 训练 18 epoch（30 太多）
batch_size    = 64
num_workers   = 8
lr            = 1e-3
hard_weight   = 3.0   # 708 hard labels 上权
positive_weight = 3.0 # 正窗口上权
mixup_alpha   = 0.4
spec_augment  = True  # Time masking + Freq masking
amp           = True  # fp16 autocast 但 mel 前端 fp32
```

### 5. 每 seed 耗时

| 阶段 | 单 seed 耗时 (T4) |
|---|---|
| 数据加载（首个 epoch cold） | ~3 min |
| 训练 18 epoch | ~50 min |
| Validation | ~3 min |
| ONNX export (fp32 + fp16) | ~1 min |
| Labeled row 推理 | ~1 min |
| **总计** | **~55-60 min** |

3 seed 约 **2.5-3h**，在 T4 单 session 时限（~9-12h）和周配额（30h）内都充裕。

### 6. 训练时要 watch 的指标

- `train_loss`：应持续下降，不应 NaN（NaN → AMP 问题，见[踩坑](#踩过的坑与修复清单)）
- `val_loss`（focal BCE vs teacher soft targets）：主要模型选择指标，比 `val_auc` 更稳
- `val_macro_auc_vs_teacher`：debug 用，可能部分 fold 因类别缺失变 NaN
- Sigmoid 输出均值：健康应在 0.05-0.15；**seed 20260101 均值 0.158 过自信**，被排除

### 7. 下载产物

训练成功后，在 Kaggle notebook 的 Output 面板下载：

```
mel_cnn_seed20260101.pt        (12 MB)
mel_cnn_seed20260101.onnx      (7 MB)
mel_cnn_seed20260101_fp16.onnx (4 MB, 提交用)
... (每 seed 一组)
cnn_probs_labeled.npy          (3 MB, 708 × 234 ensemble probs on labeled rows)
cnn_manifest.json              (训练元信息)
```

放到本地 `artifacts/cnn_bundle/` 下。

### 8. Alpha 扫描 + 打包

```bash
python scripts/08_cnn_alpha_sweep.py \
    --cnn-probs artifacts/cnn_bundle/cnn_probs_labeled.npy \
    --write --pick by_site

python scripts/06_package_submission.py \
    --tag v4 \
    --include-student \
    --alpha-probe 0 --alpha-temp 0 \
    --alpha-cnn 0.5
```

### 9. 本地 ONNX 导出的恢复路径

如果 Kaggle 训练跑到 ONNX export 就 crash（`onnxscript` 之类的包缺失），用这个：

```bash
python scripts/09_export_cnn_to_onnx.py
# 从 artifacts/cnn_bundle/*.pt 本地导出 fp32/fp16 ONNX + 重新跑 labeled row 推理
```

---

## Code Competition 提交流程

### 1. 保证 dataset 是最新的

```bash
python scripts/06_package_submission.py --tag v2 --include-student
kaggle datasets version -p artifacts/submission_bundle/ --dir-mode zip -m "v2"
# 等 ~30s 再重开 Kaggle notebook 网页确保看到新版本
```

### 2. Submit notebook 的 Data 挂载

- `birdclef-2026` competition
- `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` model
- `<you>/birdclef-2026` dataset（bundle）
- `ashok205/tf-wheels` notebook（TF 2.20 wheel 源）

### 3. Save Version → Save & Run All

- **Run with GPU？选 CPU**。虽然 Competition 规则允许 GPU，但我们用不上（Perch CPU 版）。
- Save & Run All 会从零开始跑 notebook，Internet=OFF 模拟提交环境。
- 成功的话 `/kaggle/working/submission.csv` 生成。

### 4. Submit to competition

Save Version 成功后，Notebook Version 页面出现 **"Submit to competition"** 按钮。点击即用该 Version 的 `submission.csv` 提交。

### 5. 查看 PB

几分钟后出现在 [leaderboard](https://www.kaggle.com/competitions/birdclef-2026/leaderboard)。Public LB 分数出现后可对比 OOF，评估 site shift 规模。

### 6. 每日限额

- Submission 无次数限制（不显示每日 5 次那种）但 Kaggle 界面最多显示近期 20 个
- `Save Version` 本身受 GPU 配额限制（CPU 不算 GPU 时间）

---

## 提交 Notebook 的运行时装环境

Kaggle Code Competition 提交运行时 **Internet=OFF**，所以任何默认没装的包必须**离线**装。

### 装 TF 2.20 for Perch v2

```python
# Cell 0
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Perch v2 用 StableHLO，TF 2.15 加载会失败，必须 2.20。

### 装 onnxruntime for CNN inference

```python
import glob as _g
_wheel_dirs = [
    '/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026/wheels',
    '/kaggle/input/birdclef-2026/wheels',
]
_ort_wheel = None
for _d in _wheel_dirs:
    _matches = sorted(_g.glob(_d + '/onnxruntime-*.whl'))
    if _matches:
        _ort_wheel = _matches[0]; break
if _ort_wheel:
    !pip install -q --no-deps {_ort_wheel}
else:
    print('WARN: onnxruntime wheel not found; CNN disabled')
```

**本地下 wheel**：

```bash
.venv/bin/python -m ensurepip
.venv/bin/python -m pip download \
    --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.12 \
    --no-deps \
    -d artifacts/submission_bundle/wheels/ \
    onnxruntime flatbuffers protobuf sympy mpmath packaging

# ⚠️ 把 numpy wheel 删掉！否则会装到比 Kaggle 内置 numpy 版本低的
rm artifacts/submission_bundle/wheels/numpy-*.whl  # 如果有
```

`--no-deps` 关键：避免引入 numpy/protobuf 强依赖冲突。onnxruntime 的间接依赖要手动 list，但大部分 Kaggle 已经内置了。

---

## 踩过的坑与修复清单

### 数据集上传

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | `wheels/` 子目录不在数据集里 | 默认 `--dir-mode=skip` | 用 `--dir-mode zip`，Kaggle 自动解压为同名目录 |
| 2 | 刚 push 完，notebook 里看到旧版本 | Kaggle CDN 缓存 | 等 30-60s + 重开 notebook |
| 3 | `kaggle datasets version` 报 409 | 当前有 pending version | 等 2-3 分钟再重试 |

### 提交 Notebook Runtime

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 4 | `ModuleNotFoundError: onnxruntime` | Kaggle 默认不装，submit 时 Internet=OFF | 离线 wheel 方案（见上） |
| 5 | `TF 2.20 load SavedModel` 失败 | Kaggle 默认 TF 2.15 不支持 StableHLO | 挂 `ashok205/tf-wheels` 装 2.20 |
| 6 | PB 0.876 但日志显示 CNN 没跑 | `onnxruntime` 缺失被 skip | 装 wheel + 确认 `CNN ensemble: N seeds loaded` |
| 7 | PB 从 0.876 降到 0.802 | `TIME_BUDGET=87min` 的早退逻辑**是对的**；但我误以为上限是 9h 就把它移除，结果让 CNN 跑满超过 90min 直接 Timeout | 保留 87min 早退 + 降低 CNN 成本；或者加更激进的 early-drop 阈值 |
| 8 | 3-seed / 2-seed / 1-seed CNN 都 Notebook Timeout | 我**误判比赛上限为 9h**（实际 **90min**！）；Perch 单模型 ~9.6s/file × ~560 files = 90min 刚好卡满，加任何 CNN 都爆 | Perch 必须先瘦身才有 CNN 预算（int8 量化 / 6 窗口采样 / 批推理并行） |
| 9 | Papermill 运行时 `print` 不出现 | 默认 stdout 缓冲 | 加 `flush=True` 或 `-u` 启动 |

### CNN 训练（Kaggle GPU）

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 10 | 训练 loss NaN | AMP fp16 + `torchaudio.MelSpectrogram` 下溢；STFT 输出近零 → `to_db` log(0) = -inf | `with torch.amp.autocast('cuda', enabled=False):` 包 mel 前端；mel 输出 `.clamp(min=1e-6)` |
| 11 | `val_macro_auc_vs_teacher = NaN` | 某 fold 某些类全 0 / 全 1 | 过滤需同时含正负样本的列；改用 `val_loss` 做模型选择 |
| 12 | ONNX export 报 `SymbolicValueError: STFT does not currently support complex types` | 老 ONNX exporter 不支持 `torch.stft` 输出复数 | `torch.onnx.export(..., dynamo=True)` 用 onnxscript 新 exporter |
| 13 | Kaggle 训练 notebook 报 `ModuleNotFoundError: onnxscript` | 默认不装 | 训练时 Internet=On 可 `pip install onnxscript`；或本地用 `scripts/09_export_cnn_to_onnx.py` 导出 |
| 14 | `torch.onnx.export(dynamo=True)` 返回 None | 新 exporter 把结果返回给调用者，不自动写文件 | 拿到 `ExportedProgram` 后 `.save(path)` |
| 15 | Pathing：找不到 distill cache | Kaggle 有时挂 `/kaggle/input/birdclef-2026-distill`，有时 `/kaggle/input/datasets/<owner>/birdclef-2026-distill` | 两个都 probe |
| 16 | GPU OOM at batch 64 | 首个 epoch 有 cold worker 开销峰值 | batch 降到 48 或 num_workers 降到 4 |
| 17 | 某 seed 训完 sigmoid 均值 0.158 (过自信) | 随机性 + 学习率噪声 | 该 seed 加入永久排除列表（本 repo: `20260101`） |

### 代码细节

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 18 | 本地 `scripts/09_export...` 报 `@dataclass AttributeError` | 动态 `importlib.util.spec_from_file_location` 加载时模块没注册到 `sys.modules`，`@dataclass` 找不到自身 | `sys.modules[name] = module` 再执行 spec.exec_module |
| 19 | `.pt` checkpoint 加载报 `KeyError: 'model'` | 保存时 key 是 `state_dict` 不是 `model` | `model.load_state_dict(ckpt['state_dict'])` |
| 20 | `ONNX load` 推理结果全 0 | fp16 ONNX 的输入被当成 fp32 喂进去 | `ort_sess.get_inputs()[0].type` 确认，转换 input dtype |
| 21 | `GroupKFold` 划分不均 | 默认按组数 ceil 分桶，有小组掉到同一 fold | 用 `n_splits = min(k, unique_groups)` 兜底 |

---

## 调试常用命令

### 本地验证 bundle 内容

```bash
.venv/bin/python -c "
import pickle
with open('artifacts/submission_bundle/submission_bundle.pkl','rb') as f: b = pickle.load(f)
print('tag:', b.get('tag'))
print('keys:', list(b.keys()))
print('fusion:', b['fusion_weights'])
print('cnn:', b.get('cnn'))
print('postprocess:', b.get('postprocess'))
print('student alpha:', b.get('student', {}).get('alpha_student'))
"
```

### 本地验证 ONNX 可用

```bash
.venv/bin/python -c "
import onnxruntime as ort, numpy as np
s = ort.InferenceSession('artifacts/cnn_bundle/mel_cnn_seed20260215_fp16.onnx', providers=['CPUExecutionProvider'])
print('input:', s.get_inputs()[0])
print('output:', s.get_outputs()[0])
x = np.random.randn(1, 160000).astype(np.float32)
p = s.run(None, {s.get_inputs()[0].name: x})[0]
print('shape:', p.shape, 'range:', p.min(), p.max())
"
```

### 查看 Kaggle 数据集最新版本

```bash
kaggle datasets status -d <you>/birdclef-2026
kaggle datasets list -s birdclef -m
```

### 对比两个 bundle 的 fusion 权重

```bash
.venv/bin/python -c "
import pickle
for tag in ['v2', 'v3', 'v4']:
    try:
        with open(f'artifacts/submission_bundle_{tag}.pkl','rb') as f: b = pickle.load(f)
        print(tag, b['fusion_weights'], 'cnn:', b.get('cnn'))
    except FileNotFoundError:
        print(tag, 'missing')
"
```

### 模拟 Kaggle 单文件推理时间

```bash
.venv/bin/python -c "
import time, numpy as np, onnxruntime as ort
s = ort.InferenceSession('artifacts/cnn_bundle/mel_cnn_seed20260215_fp16.onnx', providers=['CPUExecutionProvider'])
x = np.random.randn(12, 160000).astype(np.float32)  # 12 windows = 1 file
for _ in range(3): s.run(None, {'wav': x})  # warmup
t0 = time.time()
for _ in range(10): s.run(None, {'wav': x})
print(f'CNN 1 seed per file: {(time.time()-t0)/10*1000:.1f} ms')
"
```

### Force-refresh Kaggle notebook dataset mount

Kaggle notebook 网页 → 右侧 Data 面板 → 数据集条目右边三点 → **Detach** → **Re-attach** → pick latest version。

### GPU 训练实时监控

训练 notebook 里加：

```python
from IPython.display import clear_output
import time
start = time.time()

for epoch in range(epochs):
    ...  # 训练一 epoch
    elapsed_min = (time.time() - start) / 60
    eta_min = elapsed_min / (epoch + 1) * (epochs - epoch - 1)
    print(f"[epoch {epoch+1}/{epochs}] loss={tr_loss:.4f} val={val_loss:.4f} "
          f"elapsed={elapsed_min:.1f}min ETA={eta_min:.1f}min", flush=True)
```

避免 `clear_output()`——Kaggle 会在 Version run 时丢掉中间日志，只保留最终输出。

---

## 附录：本 repo 里 Kaggle 相关的文件

- `kaggle_submit/submit.ipynb` — 提交推理 notebook（CPU，加载 bundle + Perch + CNN）
- `kaggle_gpu/train_mel_cnn.ipynb` — GPU 训练 notebook（T4，MobileNetV3-Small mel-CNN）
- `scripts/06_package_submission.py` — 打包 bundle + `dataset-metadata.json`
- `scripts/07_prepare_cnn_distill.py` — 打包 GPU 训练输入
- `artifacts/submission_bundle/dataset-metadata.json` — Kaggle 主 dataset 元信息
- `artifacts/submission_bundle/wheels/*.whl` — `onnxruntime` 离线 wheel 源
- `requirements.txt` — 本地训练依赖（TF 2.20、PyTorch、torchaudio、onnxruntime、sklearn）
