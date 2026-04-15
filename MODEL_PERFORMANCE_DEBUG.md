## 模型性能排查记录（持续更新）

本文档用于记录 **HCP-Fusion Net** 在本数据集上的训练/验证/测试表现、排查过程、实验对照与结论沉淀。  
后续任何新的排查与实验结果，都只在本文档中追加更新，避免信息分散。

---

## 0. 背景与目标

- **任务**：PTMC_LNM（多模态超声 + 临床特征）预测  
  - LNM（整体转移）
  - CentralLNM（中央区转移）
  - LateralLNM（侧方区转移）
- **当前重点问题**：单任务（CentralLNM）出现 **Train 很好但 Val/Test 很差**，疑似泛化失败。
- **硬件/环境（关键约束）**
  - GPU：NVIDIA TITAN V 12GB ×4
  - Driver：470.86（CUDA Driver 11.4）
  - PyTorch：曾存在 `2.4.1+cu121` 与驱动不兼容；后切换为可用的 CUDA 11.x 版本（当前训练可正常用 GPU）。

---

## 1. 数据划分与泄漏检查（train/val/test）

我们对 `groups/train.txt, groups/val.txt, groups/test.txt` 做了统计，规则：
- **样本名**：以 `stem` 为准（去掉 `.jpg` 等后缀）
- **患者ID**：文件名第一个下划线 `_` 前的数字串（例如 `47495071_2_1_1` → patient_id=`47495071`）

### 1.1 Split 基本信息

| split | 样本数 | 唯一名称数 | 患者数 |
|---|---:|---:|---:|
| train | 577 | 577 | 258 |
| val | 188 | 188 | 86 |
| test | 205 | 205 | 87 |

### 1.2 同名样本跨 split 重复（应为 0）

| 交集 | 重复数 |
|---|---:|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

### 1.3 患者泄漏（同患者跨 split，理想应为 0）

| 交集 | 泄漏患者数 |
|---|---:|
| patient(train) ∩ patient(val) | 0 |
| patient(train) ∩ patient(test) | 0 |
| patient(val) ∩ patient(test) | 0 |

### 1.4 各 split 标签分布（按任务）

| split | task | pos | neg | pos_rate | missing |
|---|---|---:|---:|---:|---:|
| train | LNM | 310 | 267 | 0.537 | 0 |
| train | CentralLNM | 308 | 269 | 0.534 | 0 |
| train | LateralLNM | 40 | 537 | 0.069 | 0 |
| val | LNM | 104 | 84 | 0.553 | 0 |
| val | CentralLNM | 100 | 88 | 0.532 | 0 |
| val | LateralLNM | 18 | 170 | 0.096 | 0 |
| test | LNM | 100 | 105 | 0.488 | 0 |
| test | CentralLNM | 98 | 107 | 0.478 | 0 |
| test | LateralLNM | 18 | 187 | 0.088 | 0 |

**结论（阶段性）**：
- 同名样本与患者泄漏均为 **0**，说明“数据泄漏”不是主要矛盾。
- CentralLNM 在 train/val/test 的阳性率相对均衡（约 0.48~0.53），但 LateralLNM 极不均衡（约 0.07~0.10）。

---

## 2. 训练日志可信度改造（避免 Train 指标虚高）

### 2.1 问题
训练时 `train_loader` 可能包含：
- WeightedRandomSampler（带放回重采样）
- 训练增强（随机翻转/旋转/颜色抖动）

因此训练过程中的 “Train 指标” 容易 **虚高**，不代表真实拟合程度。

### 2.2 改造
在训练脚本中增加：
- `train_eval_loader`：使用 **原始 train 集**（不加权采样、关闭增强）进行评估
- 日志同时打印：
  - `Train (training batches)`：loss 走势（来自训练 loader）
  - `Train (eval on original train set)`：更可信的训练集指标
  - `Val`：验证集指标

**结论（阶段性）**：
即便用“原始 train 集评估”，Central 单任务仍出现 Train AUC 接近 1，但 Val AUC ~0.58 左右 → 过拟合/泛化失败确实存在，并非 sampler 造成的假象。

---

## 3. CentralLNM 单任务：对照实验（A/B/C）

为了定位“泛化差”的来源，我们做了三条对照命令（只列核心变更点与结果）。

> 说明：以下指标均来自日志中最优 epoch（由 AUC_CentralLNM 监控）。

### 3.1 实验 A：强正则 + 小 backbone（ResNet18 + 冻结 + BCE）

| 指标 | Train (eval) | Val | Test |
|---|---:|---:|---:|
| AUC_CentralLNM | 0.9294 | 0.6105 | 0.5306 |
| Acc_CentralLNM | 0.8368 | 0.5691 | 0.5220 |

**解读**：压住了训练集拟合（Train AUC 不再接近 1），但 Val/Test 依然弱 → “容量过大”不是唯一矛盾。

### 3.2 实验 B：去掉临床特征（use_clinical=false）

| 指标 | Train (eval) | Val | Test |
|---|---:|---:|---:|
| AUC_CentralLNM | 0.9939 | 0.5597 | 0.5569 |
| Acc_CentralLNM | 0.9705 | 0.5372 | 0.5317 |

**解读**：去掉临床并未显著提升泛化（甚至 Val 更差），说明临床分支不是主要单点问题。

### 3.3 实验 C：仅用 B 模态（modalities=B）

| 指标 | Train (eval) | Val | Test |
|---|---:|---:|---:|
| AUC_CentralLNM | 0.9850 | 0.6393 | 0.5099 |
| Acc_CentralLNM | 0.9080 | 0.5372 | 0.5122 |

**解读**：Val 有所提升，但 Test 仍接近随机 → val/test 之间也可能存在差异，或整体信号弱且不稳定。

---

## 4. 患者级 5-fold CV（CentralLNM）

我们实现了患者级 5 折交叉验证，避免单次划分偶然性。每折 best val AUC：

| fold | best_val_auc |
|---:|---:|
| 0 | 0.6453 |
| 1 | 0.6261 |
| 2 | 0.5545 |
| 3 | 0.5956 |
| 4 | 0.5344 |

**结论（阶段性）**：
- 每折 best val AUC 均偏低且波动较大（0.53~0.65），说明问题具有一致性：不是某一折运气差。

---

## 5. 患者级聚合评估（max / mean / top-k mean）

目的：验证“图像级监督 vs 患者级标签”粒度不匹配是否是主要原因。  
方法：对每折 test，先按图像推理得到概率，再按患者ID聚合为患者级概率，计算 AUC。

### 5.1 每折 AUC（patient-level）

| fold | n_patients | n_images | AUC(max) | AUC(mean) | AUC(top3-mean) |
|---:|---:|---:|---:|---:|---:|
| 0 | 87 | 183 | 0.5601 | 0.5457 | 0.5426 |
| 1 | 87 | 181 | 0.6356 | 0.6457 | 0.6378 |
| 2 | 86 | 210 | 0.5516 | 0.5532 | 0.5276 |
| 3 | 86 | 208 | 0.5330 | 0.5516 | 0.5461 |
| 4 | 85 | 188 | 0.5095 | 0.5407 | 0.5245 |

**阶段性结论**：
- 患者级聚合并未显著提升整体表现（多数折仍约 0.54~0.55），仅 fold1 较好（~0.65）。
- 说明“仅靠后处理聚合”不足以解决泛化问题；更可能存在 **域差异/标签噪声/信号弱** 等结构性原因。

---

## 6. 当前阶段结论（可执行）

### 6.1 已排除/基本排除
- **数据泄漏**（同名样本重复、患者跨 split）→ 均为 0
- **Train 指标虚高仅由 sampler 导致** → 在原始 train 集评估后仍然过拟合

### 6.2 仍需重点验证的根因（下一阶段排查）
- **域差异（train vs val/test 可分性）**：建议做一个 “train vs test 判别器/embedding 可分性” 诊断
- **标签噪声/一致性问题**：建议对“高置信错例”做医生复核
- **监督粒度不匹配**：患者级聚合提升有限，下一步应考虑 MIL（患者=bag）训练方式验证
- **临床-only baseline**：用传统模型（LogReg/XGBoost）做患者级 5-fold，判断数据可分性上限

---

## 8. 临床-only 患者级 baseline（LogReg，CentralLNM）

目的：判断 **仅靠临床特征** 在患者级预测 CentralLNM 的可分性上限，并作为“深度模型应达到的下限/参考基线”。

### 8.1 方法

- **输入**：`clinical_features.csv`（45维，无表头）  
- **对齐**：`lnm_clinical_flag.txt`（样本名 → CSV 行号）  
- **患者级标签**：同一患者任一切面 `CentralLNM=1` 则患者为 1  
- **聚合策略（患者级特征）**：
  - `mean`
  - `max`
  - `mean+max`（拼接 90 维）
- **模型**：`StandardScaler + LogisticRegression(class_weight=balanced)`  
  - 用 **val 集选择 C**（0.01/0.1/1/3/10），在 test 集报告 AUC

### 8.2 结果（每折 test AUC）

| fold | AUC_test(mean) | AUC_test(max) | AUC_test(mean+max) |
|---:|---:|---:|---:|
| 0 | 0.7197 | 0.7154 | 0.7170 |
| 1 | 0.7160 | 0.7191 | 0.7239 |
| 2 | 0.6536 | 0.6514 | 0.6454 |
| 3 | 0.6787 | 0.6939 | 0.6896 |
| 4 | 0.6867 | 0.6918 | 0.6856 |

**解读（阶段性）**：
- 临床-only baseline 在多数折能达到 **~0.65–0.72** 的患者级 AUC，明显高于深度图像模型（~0.53–0.65）。
- 说明 CentralLNM 并非“完全不可分”，至少临床特征中存在稳定信号。
- 也说明当前深度图像模型泛化失败，可能来自：
  - 图像分支学到的信号不稳定/域差异大
  - 融合策略导致图像噪声干扰临床信号（多模态+大模型更易过拟合捷径）

### 8.3 下一步建议（优先级）

1. **把深度模型的目标改为“至少不拖累临床 baseline”**  
   - 强烈建议先做：`clinical-only`、`image-only`、`clinical+deep-rad-score` 的三路对照，定位“图像到底带来的是增益还是噪声”。
2. **在融合上做“late fusion gating/加权”**（即便不改 backbone，也可先在融合 MLP 前做权重调制），避免图像噪声覆盖临床信号。
3. 若确认图像信号确实弱且不稳定：考虑 **仅用临床特征** 或以临床为主、图像为辅的策略作为临床转化版本。

---

## 9. 三路对照（CentralLNM，患者级 5-fold，待补全）

本节目标：用同一套患者级 5-fold 划分，对比三条路线的患者级 test AUC，判断“图像是增益还是噪声”。

### 9.1 三条路线定义

| 路线 | 输入 | 训练方式 | 输出评估 |
|---|---|---|---|
| Clinical-only | 45维临床特征（患者级聚合） | LogisticRegression（val 选 C） | patient-level AUC（test） |
| Image-only (B) | 仅 B 模态图像 | 深度模型训练（train.py，fold_k） | patient-level AUC（max/mean/topk） |
| Clinical + DeepRadScore | 临床特征 + DeepRadScore（从 image-only 提取） | LogisticRegression（val 选 C） | patient-level AUC（test） |

### 9.2 已完成

- Clinical-only：见 **第 8 节**（患者级 test AUC 多数折 ~0.65–0.72）

- Image-only (B)（中央任务，患者级聚合：max/mean/top3-mean）
  
  评估来源：`checkpoints/cvCentral_Bonly_r50_patientAgg.csv`

  | fold | AUC(max) | AUC(mean) | AUC(top3-mean) |
  |---:|---:|---:|---:|
  | 0 | 0.5734 | 0.5766 | 0.5739 |
  | 1 | 0.6207 | 0.6293 | 0.6202 |
  | 2 | 0.5925 | 0.6077 | 0.5968 |
  | 3 | 0.4790 | 0.5390 | 0.5352 |
  | 4 | 0.5686 | 0.5741 | 0.5546 |

  **阶段性结论**：图像-only 的 patient-level AUC 整体在 **~0.57–0.63**，且波动较大；多数折远低于临床-only（~0.65–0.72）。

- Clinical + DeepRadScore（Central 单任务，patient embedding→PCA1 DeepRadScore；临床聚合默认 `mean+max`）
  
  评估来源：`checkpoints/cvCentral_Bonly_r50_clinPlusDeepRad_cv.csv`

  | fold | AUC_test（clinical_agg=mean+max） |
  |---:|---:|
  | 0 | 0.7101 |
  | 1 | 0.7229 |
  | 2 | 0.6448 |
  | 3 | 0.6476 |
  | 4 | 0.6918 |

  **阶段性结论**：加入 DeepRadScore 后，patient-level AUC 主要与临床-only **接近**，并非稳定明显提升（例如 fold3 反而较 clinical-only 的约 0.6896 更低）。

### 9.3 本阶段对“先前问题”的判断

结合三路对照的客观结果：

1. **图像-only 的泛化能力偏弱**（多数折 patient-level AUC ~0.57–0.62），说明图像分支学到的有效信号对跨分割不稳，容易出现你观察到的“Train 高但 Val/Test 低”的现象。
2. **临床-only 具有更稳定更强的可分性**（patient-level AUC ~0.65–0.72），因此最终性能主要受临床信号驱动。
3. **临床 + DeepRadScore 并未稳定提升**：DeepRadScore 作为“图像贡献”在某些折会变好，但并不形成稳定增益，提示当前图像特征与任务标签之间存在“噪声/域差异/信息弱”的限制。

因此，先前你遇到的“训练集越来越好但验证集很差”的根因，更可能是：
- 深度图像分支学到的关联是**不稳定或域敏感**的；
- 在融合时图像分支可能会**干扰/稀释**临床信号（即使你在网络结构上做了 late fusion）。

---

## 7. 变更记录（代码层）

为支持上述排查与实验，已完成的关键代码修改/新增脚本：
- **修复单任务 key 映射**
  - `networks.py`: `task_mode=single` 时根据 `opt.tasks[-1]` 输出对应 key（lnm/central/lateral）
  - `metrics.py`: `task_mode=single` 时根据 `opt.tasks[-1]` 选择正确的 active_keys
- **训练日志可信度增强**
  - `dataset.py`: 增加 `eval_mode`（关闭增强、禁用 sampler）
  - `train.py`: 增加 `train_eval_loader` 并打印 train_eval 指标 + 训练 loss
- **排查脚本**
  - `HCP-Fusion Net/split_stats.py`
  - `HCP-Fusion Net/cv_5fold_central.py`
  - `HCP-Fusion Net/patient_level_agg_eval.py`

