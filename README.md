# BirdCLEF+ 2026 — Perch Distillation Pipeline

端到端的 [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) (Pantanal, Brazil) 工程实现：本地 Mac (M3/M4, MPS) 训练 + Kaggle CPU 推理。训练生成一个 artifact bundle，Kaggle notebook 加载后跑 Perch + fusion + post-processing。

> **最终战绩 / 状态**：Public LB **0.883 (v2)**。v3 (MLP probe + self-attn) OOF 更高但 PB 倒退到 0.835（site-shift 过拟合）。v4 mel-CNN 因 Kaggle 时限问题所有变体 (3/2/1 seed) 均 Notebook Timeout，未能获得 PB 分数。详见下方[战绩复盘](#战绩复盘)。

> 👉 **[KAGGLE_NOTES.md](./KAGGLE_NOTES.md)** — Kaggle 平台对接完整参考：比赛信息、平台约束、数据集上传坑、GPU 训练流程、Code Competition 提交、21 个踩坑清单 + 调试命令。查 Kaggle 相关任何问题都看这里。

---

## 目录结构

```
.
├── configs/base.yaml            # 路径、超参、fusion 权重统一配置
├── common/                      # 共享模块（sys.path 导入，非 pip install）
│   ├── paths.py                 # 本地 / Kaggle 路径自动切换
│   ├── filenames.py             # BC2026_* filename -> site / hour / month
│   ├── taxonomy.py              # primary_labels, Perch 标签映射 + proxy
│   ├── audio.py                 # OGG 读取 + 12 × 5s 窗口
│   ├── perch.py                 # Perch v2 SavedModel 包装（批推理）
│   ├── priors.py                # Fold-safe 先验表（global/site/hour/month/site×hour）
│   ├── features.py              # StandardScaler + PCA + 时间统计
│   ├── probes.py                # 线性 probe (LogReg)
│   ├── probes_mlp.py            # MLP probe (v3 heads)
│   ├── temporal_attn.py         # 1-layer 4-head self-attn temporal head (v3 heads)
│   ├── student.py               # MLP student (PyTorch + MPS)
│   ├── pseudo.py                # 伪标签筛选/阈值
│   ├── fusion.py                # 加权 logit fusion
│   ├── postproc.py              # TopN 平滑 + isotonic 校准
│   ├── metrics.py               # Macro ROC-AUC
│   └── io_utils.py              # pickle/npz/json helper
├── scripts/                     # 本地流水线
│   ├── 01_build_perch_cache.py      # Perch embedding + logit 缓存
│   ├── 02_fit_teacher.py            # v1 teacher (线性 probe + temp-lite) + GroupKFold OOF
│   ├── 02b_fusion_sweep.py          # OOF fusion 权重网格搜索
│   ├── 02c_pseudo_iter.py           # 迭代伪标签（生成 ~60k pseudo-labels）
│   ├── 02d_fit_v3_heads.py          # 训练 MLP probe + self-attn temp 头（v3 heads）
│   ├── 02e_site_oof_diag.py         # GroupKFold-by-site 诊断（发现 domain shift）
│   ├── 03_train_student.py          # MLP student 5-fold OOF
│   ├── 04_calibrate.py              # isotonic vs TopN A/B 测试
│   ├── 04b_per_class_delta.py       # 每类 logit 偏移（当前 0，占位）
│   ├── 05_local_oof_eval.py         # 本地 OOF macro AUC sanity check
│   ├── 06_package_submission.py     # 打包 bundle → artifacts/submission_bundle/
│   ├── 07_prepare_cnn_distill.py    # 准备 GPU 训练用的 distill cache
│   ├── 08_cnn_alpha_sweep.py        # CNN alpha 扫描（labeled rows）
│   └── 09_export_cnn_to_onnx.py     # 本地 .pt → ONNX 导出 + labeled row 推理
├── kaggle_submit/submit.ipynb   # Kaggle CPU 推理 notebook
├── kaggle_gpu/train_mel_cnn.ipynb   # Kaggle T4 GPU CNN 训练 notebook
├── legacy/notebooks/            # 参考 notebook（原始 + 2nd-place 复盘 ipynb）
├── logs/                        # 训练运行日志（gitignored）
├── data/birdclef-2026/          # 比赛数据（gitignored）
└── models/perch_v2_cpu/1/       # Perch v2 SavedModel（gitignored）
```

Gitignore：`data/`、`artifacts/`、`models/perch_v2_cpu/`、`.venv/`、`__pycache__/`、`logs/`、`*.pkl/*.npy/*.parquet`。

---

## 战绩复盘

### Kaggle Public LB 演进

⚠️ **比赛的实际 CPU 提交上限是 90 分钟**（不是我最初以为的 9h）。这个误判是 v4 系列全部 timeout 的根因——详见下方 [时限误判的代价](#时限误判的代价)。

| 版本 | 策略（与前一版的 delta） | 本地 OOF (by-file / by-site) | Kaggle PB | 备注 |
|---|---|---|---|---|
| v1 | Perch + Prior + 线性 probe + 线性 temporal | — | **0.693** | 90min CPU 工程 baseline |
| v2 | + 5-fold student MLP (α=0.4) + TopN=1 file-level smoothing | **0.9239** / 0.8252 | **0.883** | 稳定可靠的基线 |
| v3 | + MLP probe (1536→512) + self-attn (1L 4H) temporal | 0.9495 / **0.8477** | **0.835** | 反而下降 → 暴露 site-shift |
| v4.0 (CNN 未生效) | v3 + 3-seed mel-CNN 集成（但 `onnxruntime` 缺失 → 实际只跑 Perch） | — | **0.876** | 意外发现：Perch 单模型 ~9.6s/file × ~560 files = 刚好 90min |
| v4.1 | wheels 修复，`onnxruntime` 可用；`TIME_BUDGET=87min` 早退仍在 | — | **0.802** | CNN 开销让早退触发，截断的行被填 0 |
| v4.2 (3-seed)   | **误移除**早退逻辑（以为上限是 9h），3-seed CNN @ α=1.5 全跑 | — / 0.9238 | **Timeout** | 110 min > 90 min |
| v4.3 (2-seed)   | drop seed 20260101，[215+322] CNN @ α=1.5 | — / 0.9333 | **Timeout** | 103 min > 90 min |
| v4.4 (1-seed)   | drop v3 heads + 单 seed 20260215 @ α=0.5 | — / 0.9151 | **Timeout** | 96.5 min > 90 min |

**最终用分：PB 0.883（v2 bundle）。**

### 关键技术发现

1. **Site domain shift 是核心瓶颈**
   - by-file OOF 0.9466 vs by-site OOF 0.8477，Δ = **-0.099**
   - 解释了 v3 OOF 0.9495 → PB 0.835 的反常下降：本地 fold 留的是 file 而非 site
   - 诊断脚本：`scripts/02e_site_oof_diag.py`

2. **v3 heads 在小 labeled 集合上过拟合了 site 分布**
   - MLP probe + self-attn 在 708 条 labeled rows（分属 8 个 site）上优于线性头
   - by-site 测评时反而不如 v2 线性头 → site-specific 模式被过拟合
   - 正确策略：drop v3 heads，用 v2 骨架 + CNN 补 site shift

3. **Iterative pseudo-labeling：对 probe 没用，对 student 有用**
   - 自蒸馏 probe：0.8946 → 0.8912（倒退）
   - 但产出的 ~60k pseudo-labels 给 student MLP 用后：0.8946 → 0.9159（+0.021）
   - 产出路径：`artifacts/pseudo_export.pkl`

4. **Isotonic 校准反而伤性能**
   - A/B: TopN=1 alone > TopN=1 + Isotonic（脚本 `04_calibrate.py`）
   - 原因：每类正样本太少（234 classes ÷ 708 rows），isotonic knots 过拟合

5. **CNN 单 seed 选择反直觉**
   - α=1.5 固定时：seed 20260322 最好 (by_site 0.9004)
   - 每个 seed 单独扫 α 后：**seed 20260215 @ α=0.5 (by_site 0.9151) > seed 20260322 @ α=1.0 (0.9053)**
   - seed 20260101 均值 0.158 过自信，拖累集成 → 永远排除
   - 2-seed [215+322] @ α=2.5 by_site 0.9333（比 3-seed 好 +0.01，省 33% 算力）

6. **训练端踩坑记录**
   - **AMP + `torchaudio.MelSpectrogram` 下溢 NaN**：STFT fp16 输出近零 → `to_db` 产生 NaN。修法：`torch.amp.autocast('cuda', enabled=False)` 包住 mel 前端
   - **ONNX export 失败**：`torch.onnx.export` 老版不支持 `torch.stft` 复数输出 → `SymbolicValueError`。修法：`dynamo=True` 启用 `onnxscript` 新 exporter
   - **Kaggle 离线环境缺 `onnxruntime`**：提交时 Internet=OFF 无法 pip install。修法：本地下载 wheels，打包进 dataset，notebook Cell 0 离线安装

### 时限误判的代价

我在实现早期把**提交上限记成 9 小时**（Kaggle 平台通用的交互式 session 上限），实际这个比赛的 Code Competition 规则明确写了 **CPU Notebook ≤ 90 minutes**。这个误判造成：

1. **v4.1 PB 回退（0.876 → 0.802）**：原有 87min 早退逻辑是**正确的防御**；但截断造成丢行补零
2. **v4.2/4.3/4.4 全部 Timeout**：以为有 9h 预算，移除早退让 CNN 全跑 → 实际都超过 90min

90 min = 5400s 的实际时限下：

| 配置 | ~s/file | × ~560 files | 占用 90min 上限 | 实测 |
|---|---|---|---|---|
| No CNN (Perch only) | 9.6 | **90.0 min** | 刚好 100% | ✅ v4.0 PB 0.876 |
| 1-seed CNN | 10.3 | 96.5 min | 107% | ❌ Timeout |
| 2-seed CNN | 11.0 | 103 min | 114% | ❌ Timeout |
| 3-seed CNN | 11.7 | 110 min | 122% | ❌ Timeout |

**结论**：想上 CNN 就必须先把 Perch 推理成本降下去。Perch 单模型已经把 90min 预算填满，任何额外头都进不来。

教训：提交前应主动查比赛"Code Requirements"板块确认硬上限，不要套用 Kaggle 其他场景的默认值。

---

## 环境搭建

### 1. Python 环境

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 竞赛数据

```bash
# 期望 data/birdclef-2026/ 包含:
# taxonomy.csv, sample_submission.csv, train.csv, train_soundscapes_labels.csv,
# train_audio/, train_soundscapes/, test_soundscapes/, recording_location.txt
```

### 3. Perch v2 模型

`models/perch_v2_cpu/1/` 需放 [Perch v2 TensorFlow SavedModel](https://www.kaggle.com/models/google/bird-vocalization-classifier)。

---

## 本地流水线

所有脚本从 `configs/base.yaml` 取路径，默认幂等（输出已存在就跳过，传 `--overwrite` 重建）。

### 基线（v2 bundle，PB 0.883）

```bash
# 1. Perch 缓存（最重，~1-3h on M3/M4 CPU）
python scripts/01_build_perch_cache.py

# 2. v1 teacher：priors + 线性 probe + temp-lite + fusion，GroupKFold-by-file OOF
python scripts/02_fit_teacher.py

# 2b. 诊断：OOF fusion 权重扫描
python scripts/02b_fusion_sweep.py

# 2c. 迭代伪标签生成（写 artifacts/pseudo_export.pkl, ~60k rows）
python scripts/02c_pseudo_iter.py

# 3. 5-fold OOF student MLP（MPS, ~5-15 min）
python scripts/03_train_student.py

# 4. Post-proc A/B：TopN=1 胜出
python scripts/04_calibrate.py

# 5. Sanity eval
python scripts/05_local_oof_eval.py

# 6. 打包 v2 bundle
python scripts/06_package_submission.py \
    --tag v2 \
    --include-student \
    --alpha-perch 0.40 --alpha-prior 0.05 \
    --alpha-probe 0.10 --alpha-temp  0.15 \
    --alpha-student 0.40
```

### v3 heads（OOF 高但 PB 低，不推荐作为最终提交）

```bash
# 训练 MLP probe + self-attn temporal head
python scripts/02d_fit_v3_heads.py

# by-site OOF 诊断（揭示 site shift）
python scripts/02e_site_oof_diag.py

# 打 v3 bundle
python scripts/06_package_submission.py --tag v3 --include-student
```

### v4 CNN（未能在 Kaggle 时限内提交成功，保留作参考）

```bash
# 7. 准备 GPU 训练用 distill cache（uint8 量化 teacher probs）
python scripts/07_prepare_cnn_distill.py
kaggle datasets version -p artifacts/cnn_distill -m "distill v2"

# 8. 到 Kaggle 跑 kaggle_gpu/train_mel_cnn.ipynb（T4 GPU, ~3h for 3 seeds）
#    下载生成的 mel_cnn_seed*_fp16.onnx + cnn_probs_labeled.npy 到 artifacts/cnn_bundle/

# 9. （可选）本地 .pt → ONNX 导出 + labeled row 推理（如果 Kaggle 导出失败时的恢复路径）
python scripts/09_export_cnn_to_onnx.py

# 10. Alpha 扫描
python scripts/08_cnn_alpha_sweep.py \
    --cnn-probs artifacts/cnn_bundle/cnn_probs_labeled.npy \
    --write --pick by_site

# 11. 打 v4 bundle
python scripts/06_package_submission.py \
    --tag v4 --include-student \
    --alpha-probe 0 --alpha-temp 0 \
    --alpha-cnn 0.5
```

---

## 上传 bundle 到 Kaggle

### 首次

```bash
# 编辑 artifacts/submission_bundle/dataset-metadata.json，把 id 换成你的 username
kaggle datasets create -p artifacts/submission_bundle/ --dir-mode zip
```

### 更新

```bash
kaggle datasets version -p artifacts/submission_bundle/ \
    --dir-mode zip -m "v2 stable baseline"
```

`--dir-mode zip` 必须开 —— 否则 `wheels/` 子目录（`onnxruntime` 离线安装用）会被跳过。Kaggle 自动解压 zip 为同名子目录。

### 在 Kaggle 跑 submit notebook

1. 上传 `kaggle_submit/submit.ipynb` 到 Kaggle
2. Data 面板挂载：
   - `birdclef-2026` competition
   - `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` model
   - `<your>/birdclef-2026` dataset（刚上传的 bundle）
   - `ashok205/tf-wheels` notebook（TF 2.20 wheels 来源）
3. Save & Run All → `/kaggle/working/submission.csv` → Submit

---

## 设计原则

- **Idempotent**：每个脚本检查输出是否存在，存在则跳过（`--overwrite` 重建）
- **Single source of truth**：`configs/base.yaml` 统一标签序、fusion 权重、pseudo 阈值
- **Fold-safe**：priors / probes / student 全部 GroupKFold OOF 训练
- **Train / infer 分离**：`common/perch.py` 离线跑缓存；Kaggle notebook 在线推理但不训练
- **路径自动**：`common/paths.py` 检测 `/kaggle/input` 并切换
- **最小提交 notebook**：fusion + post-proc 为主，CPU 时限敏感

---

## 未来工作

**90 分钟 CPU 时限是核心约束**。Perch 单模型已经吃满 100%，任何增量必须来自 Perch 自身压缩或完全替换。

1. **降 Perch 推理成本**（解锁 CNN 预算的唯一路径）
   - **int8 量化 Perch SavedModel**：TF TFLite 转换或 ORT 量化工具，目标 2-3x 加速
   - **每 file 只跑 6 个窗口**（间隔采样，相邻窗口用 teacher 插值）：直接 2x 加速
   - **线程池并行 Perch + CNN**：ORT 和 TF 推理都释放 GIL，`ThreadPoolExecutor(max_workers=2)` 可 overlap
   - **Batch size 调优**：当前 16 file/batch，试 24 或 32 看 amortization
   - **ONNX 化 Perch**（StableHLO → ONNX）：抛弃 TF 运行时开销

2. **保留 87min 安全早退 + 更精细 CNN 降级**
   - 接受 CNN 可能跑不全；前 50 个 file 计时外推 ETA，发现要超就 CNN 阶段性 drop
   - 当前的 dynamic auto-drop 思路没错，是我误判时限把它拆了

3. **绕过 site shift 但不加推理成本**
   - Site-adversarial domain adaptation（训练时对抗，推理时免费）
   - Test-time prior calibration（用 test 批次统计反推先验）
   - 把 train_soundscapes 全部 23 个 site 的 segment 纳入 pseudo-label 池（当前只用了 labeled 的 8 个 site）

4. **提交前的必做检查**（避免这次踩的坑）
   - 打开比赛 Rules → Code Requirements，确认 CPU/GPU 时限、submission 格式
   - Sanity Run 时只跑 1 个 test file，外推 `total_time = per_file × n_files` 对比上限
   - 保留 `TIME_BUDGET` 早退逻辑作为安全网

---

## 参考

- [BirdCLEF+ 2025 — 2nd place (Sydorskyy & Gonçalves)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)，CEUR [Vol-4038/paper_256](https://ceur-ws.org/Vol-4038/paper_256.pdf)。TopN 平滑、迭代伪标签的思路来自这里。
- [Perch v2 — Google Bird Vocalization Classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- 参考 notebook（`legacy/notebooks/`）：`pantanal-distill-birdclef2026-improvement-a4dc68.ipynb`（历史 0.924 PB）
