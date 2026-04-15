## 项目说明：PTMC_LNM 多模态 HCP-Fusion Net

这个项目是一个**多模态超声 + 临床特征**的深度学习框架，用来预测甲状腺乳头状癌淋巴结转移（LNM、CentralLNM、LateralLNM）。

核心组件：
- **`MultiModalDataset`**：按约定的目录结构读取多模态图像和多任务标签（见下文“数据准备”）。
- **`ClinicalCSVLoader`**：读取已经预处理好的临床特征 CSV。
- **`HCP-Fusion Net` 模型**：在 `models/` 目录下，通过 `build_model` 构建。
- **训练主脚本**：`HCP-Fusion Net/train.py`。

---

## 一、数据准备：MultiModalDataset 目录结构

默认的数据根目录（可通过 `--data_root` 修改）在：

```text
/data2/wjb/code/PTMC_LNM/data/MultiModalDataset
```

目录内部结构必须满足 `MultiModalDataset` 的要求：

```text
MultiModalDataset/
├── B/                 # B‑mode 图像
├── VUSE/              # VUSE 图像
├── VCEUS/             # VCEUS 图像
├── LNM/               # 任务1标签，每个样本一个同名 .txt，内容为 0 或 1
├── CentralLNM/        # 任务2标签
├── LateralLNM/        # 任务3标签
├── groups/
│   ├── train.txt      # 训练集样本名列表（每行一个，不带扩展名）
│   └── test.txt       # 测试/验证集样本名列表
├── lnm_clinical_flag.txt   # 图像文件名 → 临床 CSV 行号 的映射
└── clinical_features.csv   # 预处理好的临床特征（已数值化、标准化）
```

关键约定：
- `train.txt` / `test.txt` 里每一行是一条样本名，例如 `00044628_1_1_1`，**不带扩展名**。
- 图像文件实际保存在 `B/`, `VUSE/`, `VCEUS/` 中，例如 `B/00044628_1_1_1.jpg`。
- 标签文件位于 `LNM/`, `CentralLNM/`, `LateralLNM/`，例如 `LNM/00044628_1_1_1.txt`，文件内容为 `0` 或 `1`。
- `lnm_clinical_flag.txt` 每行是一个图像文件名（可带扩展名），行号 = `clinical_features.csv` 中的行索引。

如果你的真实数据不在默认位置，请把 `--data_root` 改成你自己的路径，并保持以上子目录结构不变。

---

## 二、快速开始：如何用 MultiModalDataset 训练 HCP-Fusion Net

在终端（命令行）中，`cd` 到项目根目录：

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net
```

最常用的完整三模态 + 级联多任务训练命令示例（已经在 `train.py` 顶部给出）：

```bash
python train.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --modalities B VUSE VCEUS \
  --fusion_type late \
  --task_mode multi_cascaded \
  --use_clinical true \
  --backbone resnet50 \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --penalty_weight 0.5 \
  --exp_name exp_full_cascaded
```

说明（用“说人话”的方式）：
- **`--data_root`**：告诉程序“我的 MultiModalDataset 数据放在哪里”，路径就是上面那棵目录树的根。
- **`--modalities`**：选择要用的图像模态；上面命令是用三种（B、VUSE、VCEUS）一起训练。
- **`--fusion_type late`**：每个模态单独过一个 backbone，再在特征层拼在一起（推荐）。
- **`--task_mode multi_cascaded`**：三任务级联，先预测 LNM / CentralLNM，再用它们的特征去帮忙预测 LateralLNM。
- **`--use_clinical true`**：把临床 CSV 一起喂给网络；如果你暂时没有临床特征，可以改成 `false`。
- **`--exp_name`**：这次实验的名字，会在 `checkpoints/exp_name` 下面自动新建文件夹保存日志和模型。

训练过程中，`MultiModalDataset` 会被 `build_dataloader` 自动调用，你**不需要**自己手动 new，只要保证数据目录结构正确即可。

---

## 三、常见变体（消融实验）

- **只用单模态 B 图像，不用临床特征，平行多任务：**

```bash
python train.py \
  --modalities B \
  --fusion_type early \
  --task_mode multi_parallel \
  --use_clinical false \
  --penalty_weight 0 \
  --exp_name exp_ablation_single_modal
```

- **只预测一个任务（例如只关心 LateralLNM）：**
  - 把 `--tasks` 改成只包含一个任务名；
  - 把 `--task_mode` 设为 `single`。

```bash
python train.py \
  --tasks LateralLNM \
  --task_mode single \
  --modalities B VUSE VCEUS \
  --use_clinical true \
  --exp_name exp_single_lateral
```

---

## 四、关键参数速查（对小白友好版本）

- **数据相关**
  - **`--data_root`**：你的数据总文件夹。
  - **`--modalities`**：用几种成像方式（1 种、2 种、3 种都可以）。
  - **`--use_clinical`**：要不要加上 CSV 里的临床信息。
- **模型结构**
  - **`--backbone`**：选哪种 CNN（`resnet50` 是一个比较稳的默认值）。
  - **`--fusion_type`**：`late` 一般比 `early` 更灵活稳定。
  - **`--task_mode`**：`single` / `multi_parallel` / `multi_cascaded` 三选一。
- **训练过程**
  - **`--epochs`**：训练多少轮。
  - **`--batch_size`**：一次塞进显卡的图片数量；显存不够就调小。
  - **`--lr`**：学习率，大多数情况 1e-4 就可以。

---

## 五、如何排查不能运行的问题（简单 Checklist）

1. **路径是否正确**：`--data_root` 指到的目录下，是否真的有 `B/`, `VUSE/`, `VCEUS/`, `groups/` 等子目录。
2. **文件名是否对得上**：`groups/train.txt` 里的样本名，是否在 `B/` 等目录中存在对应图片。
3. **临床特征文件**：
   - 如果 `--use_clinical true`，检查 `clinical_features.csv` 和 `lnm_clinical_flag.txt` 是否存在。
   - 如果你暂时还没有临床数据，可以先把 `--use_clinical` 设为 `false`。
4. **显存不够**：适当减小 `--batch_size`，或把 `--modalities` 改少一点。

---

## 六、本次改动与后续改进想法

本版本新增了项目总览和“说人话”的训练指南：
- 统一描述了 `MultiModalDataset` 所需的数据目录结构。
- 给出了直接可运行的训练命令示例，方便你一步到位开始训练。

同时新增了模型性能排查文档：
- `MODEL_PERFORMANCE_DEBUG.md`：汇总训练/验证异常、split 检查、对照实验与交叉验证结果（持续更新）。

后续可以在这里继续补充：
- 各指标（AUC、ACC 等）的含义解释；
- 不同模态组合、不同任务模式下的推荐配置和经验总结。


---

## 七、传统影像组学 + 传统机器学习对照（新增）

为了验证“非深度学习路线”是否对 CentralLNM 也有效，新增脚本：

- `HCP-Fusion Net/traditional_radiomics_cv5_central.py`

该脚本在同一套患者级 5-fold 划分下，统一比较三条 pipeline：

- `clinical_only`：仅临床特征
- `radiomics_only`：仅传统影像组学特征（从 B/VUSE/VCEUS 单模态图像提取统计特征）
- `clinical_radiomics`：临床 + 影像组学拼接

并对比多种模型：

- LogisticRegression
- RandomForest
- SVM(RBF)
- AdaBoost
- ExtraTrees
- XGBoost（若环境有 `xgboost` 自动启用，否则自动跳过）
- Stacking（先进集成学习）
- SoftVoting（软投票集成）

### 1) 直接运行示例

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net
python traditional_radiomics_cv5_central.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --groups_root groups_cv5_central \
  --folds 5 \
  --modality B \
  --clinical_agg mean+max \
  --rad_agg mean+max \
  --seed 42 \
  --save_dir ./checkpoints \
  --exp_prefix cvCentral_traditionalML
```

### 2) 结果输出（会自动生成）

- 明细指标表：`<save_dir>/<exp_prefix>_all_metrics.csv` / `.json`
  - 包含 train/val/test 的 AUC、ACC、Precision、Recall、F1、Sensitivity、Specificity、混淆矩阵
- 统计汇总表：`<save_dir>/<exp_prefix>_summary_mean_std.csv`
  - 按 pipeline + 模型汇总 5-fold test 的 mean±std
- 可视化图：
  - `ROC` 对比图：`<save_dir>/<exp_prefix>_roc_<pipeline>.png`
  - `AUC` 箱线图：`<save_dir>/<exp_prefix>_auc_boxplot_by_pipeline.png`

### 3) 注意事项

- 该影像组学实现是“无分割 mask 的传统统计特征基线”，优点是能直接用你当前数据快速验证传统 ML 路线。
- 如果后续你有可靠病灶分割（ROI），可在此脚本基础上升级为标准 IBSI 风格 radiomics（如 PyRadiomics 全量纹理特征）。

---

## 八、三源融合实验（临床 + 深度学习 + ROI Radiomics，新增）

这部分是你当前要推进的“完整逻辑链”版本，和上面的“无 mask 统计特征基线”不同。

### 1) 先说结论：哪些结果能复用，哪些不能

- `Ours V1 (Deep MIL + Clinical Gated, EffNet-B0)`：
  - 如果你接受当前 `v2_mil_gated/checkpoints_v2_backbones/backbone_efficientnet_b0` 作为“统一协议下的深度基线”，它可以先作为参考复用。
  - 但如果你希望和三源融合 `Ours V2` 完全共享同一套实验目录、命名和导出中间特征，建议在新目录下重跑一遍。
- `Baseline C / D (Radiomics-only / Clinical+Radiomics)`：
  - **不能**直接复用旧的 `traditional_radiomics_cv5_central.py` 结果。
  - 原因：旧脚本是“无分割 mask 的统计特征基线”，而你现在要的是“基于 ROI 的标准 PyRadiomics”。

### 2) 新增脚本

- `HCP-Fusion Net/roi_radiomics_cv5_central.py`
  - 功能：ROI 切片级 PyRadiomics 提取、患者级 mean/max 聚合、仅在训练集内做 LASSO 选择、导出 `Rad-Score / 低维 H_rad`，并完成 `Radiomics-only / Clinical+Radiomics` 两条对照。
- `HCP-Fusion Net/v2_mil_gated/run_tri_source_experiments.py`
  - 功能：串联 `ROI radiomics -> Ours V1 -> Ours V2`。

### 3) 运行前依赖

当前 `pumc` 环境默认**没有**：

- `radiomics`
- `SimpleITK`

建议先在你的 `pumc` 环境中手动安装：

```bash
conda activate /data2/wjb/anaconda3/envs/pumc
pip install pyradiomics SimpleITK
```

如果安装慢或报编译问题，也可以先试：

```bash
conda activate /data2/wjb/anaconda3/envs/pumc
conda install -c simpleitk simpleitk -y
pip install pyradiomics
```

### 4) 先跑 ROI Radiomics

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net
/data2/wjb/anaconda3/envs/pumc/bin/python roi_radiomics_cv5_central.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --mask_root /data2/wjb/code/PTMC_LNM/data/masks \
  --groups_root groups_cv5_central \
  --modalities B \
  --save_dir ./checkpoints_roi_radiomics \
  --exp_name central_roi_radiomics_cv5
```

输出重点：

- `fold_k/patient_features_selected.csv`
  - 这是后续 `Ours V2` 会直接读取的患者级组学特征文件
- `summary_mean_std.csv`
  - 对应 `Baseline C / D`

### 5) 跑深度基线 Ours V1

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net/v2_mil_gated
/data2/wjb/anaconda3/envs/pumc/bin/python train.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --groups_root groups_cv5_central \
  --save_dir ./checkpoints_v2_tri_source \
  --exp_name ours_v1_effb0 \
  --model_mode attention_mil_gated \
  --backbone efficientnet_b0 \
  --pretrained False \
  --modalities B VUSE VCEUS
```

### 6) 跑三源融合 Ours V2

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net/v2_mil_gated
/data2/wjb/anaconda3/envs/pumc/bin/python train.py \
  --data_root /data2/wjb/code/PTMC_LNM/data/MultiModalDataset \
  --groups_root groups_cv5_central \
  --save_dir ./checkpoints_v2_tri_source \
  --exp_name ours_v2_tri_source_effb0 \
  --model_mode attention_mil_gated \
  --backbone efficientnet_b0 \
  --pretrained False \
  --modalities B VUSE VCEUS \
  --use_radiomics True \
  --radiomics_root /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net/checkpoints_roi_radiomics/central_roi_radiomics_cv5
```

### 7) 一键串联

```bash
cd /data2/wjb/code/PTMC_LNM/HCP-Fusion\ Net/v2_mil_gated
/data2/wjb/anaconda3/envs/pumc/bin/python run_tri_source_experiments.py \
  --python /data2/wjb/anaconda3/envs/pumc/bin/python \
  --run_radiomics \
  --run_v1 \
  --run_v2
```

### 8) 当前三源设计的结构说明

- 深度分支：`Attention-MIL -> H_deep`
- 组学分支：`ROI PyRadiomics -> LASSO -> H_rad / Rad-Score`
- 影像混合分支：`H_img_mix = [H_deep, H_rad] -> MLP`
- 最终融合：`H_final = H_clin + (G ⊙ H_img_mix)`

这样做的好处是：

- 不破坏你已有的 `Clinical-First Gated Fusion` 主结构
- 传统组学被当作“影像侧补充证据”，而不是与临床直接粗暴拼接
- LASSO 严格在训练集内拟合，可避免 radiomics 常见的数据泄漏

### 9) 预训练+解冻策略下的 V1/V2 三骨干归档对比（新增）

统一协议：

- `pretrained=True`
- `freeze_backbone_epochs=3`（前 3 epoch 冻结，随后解冻微调）
- 患者级 `groups_cv5_central`，`attention_mil_gated`，5-fold test 汇总

已归档实验目录：

- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v1_effb0`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_effb0`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v1_swin_t_pretrained_unfreeze`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_swin_t_pretrained_unfreeze`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v1_vit_b_16_pretrained_unfreeze`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_vit_b_16_pretrained_unfreeze`

结果汇总（`summary.csv` 的 `mean±std`）：

| 组别　　　　| AUC_central   | AUC_lnm       | ACC           | F1            | Sen           | Spe           | Compliance    |
| -------------| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:|
| EffB0 V1　　| 0.6732±0.0236 | 0.6992±0.0110 | 0.6173±0.0352 | 0.6592±0.0413 | 0.7014±0.1558 | 0.5178±0.2449 | 0.9700±0.0382 |
| EffB0 V2　　| 0.6708±0.0544 | 0.7032±0.0388 | 0.6426±0.0393 | 0.6826±0.0364 | 0.7096±0.0555 | 0.5632±0.0684 | 0.9603±0.0311 |
| Swin-T V1　 | 0.6939±0.0267 | 0.7011±0.0203 | 0.6473±0.0253 | 0.6620±0.0667 | 0.6635±0.1617 | 0.6296±0.1890 | 0.9977±0.0047 |
| Swin-T V2　 | 0.6793±0.0313 | 0.6914±0.0343 | 0.6263±0.0244 | 0.6601±0.0267 | 0.6709±0.0569 | 0.5736±0.0731 | 0.9348±0.0853 |
| ViT-B/16 V1 | 0.7104±0.0253 | 0.7222±0.0295 | 0.6289±0.0345 | 0.6686±0.0411 | 0.6969±0.0956 | 0.5490±0.1152 | 0.9512±0.0626 |
| ViT-B/16 V2 | 0.6404±0.0496 | 0.6592±0.0522 | 0.5779±0.0370 | 0.6478±0.0322 | 0.7220±0.1123 | 0.4077±0.1766 | 0.9398±0.0621 |

简要观察：

- V1 阶段：`ViT-B/16` 的 AUC 最高，`Swin-T` 的规则一致性（Compliance）与特异度（Spe）更突出。
- V2 阶段：`EffB0` 相对最稳（AUC_central 基本持平，ACC/F1/Sen/Spe 提升）；`Swin-T` 与 `ViT-B/16` 出现不同程度退化，尤其 `ViT-B/16` 下降明显。
- 当前结论：V2 的主要风险不一定来自“是否使用组学”，更可能来自“深度表征与组学表征的融合方式不匹配”。

### 10) V2 退化的可能原因与改进路线（新增）

当前假设：深度特征维度高、组学分支维度低（甚至压缩为 1 维 Rad-score）时，直接拼接/加权融合会造成信息不对等，导致融合头学习不稳定。

#### 方案一：特征空间对齐投影（Feature Alignment & Projection，优先尝试）

核心思想：

- 不直接使用 1 维 `Rad-score`，改为使用 LASSO 保留的 `K` 维组学向量 `V_rad`（例如 `K=20`）。
- 用两个独立 MLP 将 `H_deep` 与 `V_rad` 投影到同一隐空间（如 128 维）：
  - `Z_deep = MLP_deep(H_deep) in R^128`
  - `Z_rad = MLP_rad(V_rad) in R^128`
- 对齐后再融合（逐元素相加或 concat+MLP），然后再交给临床门控：
  - `H_final = H_clin + (G ⊙ (Z_deep + Z_rad))`

预期优点：

- 解决“维度和尺度不对等”问题；
- 让组学信息和深度信息在可比语义空间内交互；
- 改动可控，保留现有 `clinical-first gated` 主框架。

#### 方案二：FiLM 特征调制（Feature-wise Linear Modulation）

核心思想：

- 将低维组学当作“调制器”而非“对等分支”：
  - `gamma = Linear_gamma(H_rad)`
  - `beta  = Linear_beta(H_rad)`
- 用组学生成的 `gamma/beta` 对深度特征做仿射调制：
  - `H_mod = gamma ⊙ H_deep + beta`

预期优点：

- 完全规避直接拼接的维度冲突；
- 参数量小、训练稳定性可能更好；
- 数学解释清晰：组学用于“激活/抑制”深度通道。

#### 下一步实验建议（按优先级）

1. 先做方案一（`K=20/32`，`proj_dim=128`）并与当前 V2 做一一对照；
2. 再做方案二 FiLM 版本，比较其在 `ViT-B/16` 和 `Swin-T` 上的恢复效果；
3. 固定同一阈值与同一 5-fold 划分，统一报告 `AUC + Sen + Spe + Compliance`，避免只看 AUC。

### 11) V1（VI）vs 原始V2 vs 新版V2（align\_proj）对比（新增）

说明：

- 这里的“VI”按实验上下文等同于 `V1`。
- 原始V2：`use_radiomics=True` + `legacy` 融合（旧版）。
- 新版V2：`use_radiomics=True` + `align_proj` + `radiomics_topk=20` + `rad_align_fusion=concat`。
- 新版 ViT 由于显存限制，使用 `max_instances=8` 完成训练（其余设置保持一致）。

#### 11.1 三向结果总表（5-fold test，mean±std）

| 模型　　 | 版本　　　　　　　　　| AUC_central       | AUC_lnm           | ACC               | F1                | Sen               | Spe               | Compliance        |
| ----------| -----------------------| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:|
| EffB0　　| V1　　　　　　　　　　| 0.6732±0.0236     | 0.6992±0.0110     | 0.6173±0.0352     | 0.6592±0.0413     | 0.7014±0.1558     | 0.5178±0.2449     | 0.9700±0.0382     |
| EffB0　　| 原始V2　　　　　　　　| 0.6708±0.0544     | 0.7032±0.0388     | 0.6426±0.0393     | 0.6826±0.0364     | 0.7096±0.0555     | 0.5632±0.0684     | 0.9603±0.0311     |
| EffB0　　| 新版V2（align\_proj） | **0.6911±0.0323** | **0.7037±0.0257** | 0.6034±0.0429     | 0.6228±0.0834     | 0.6290±0.1558     | **0.5744±0.1028** | 0.9608±0.0408     |
| Swin-T　 | V1　　　　　　　　　　| **0.6939±0.0267** | 0.7011±0.0203     | **0.6473±0.0253** | **0.6620±0.0667** | **0.6635±0.1617** | **0.6296±0.1890** | **0.9977±0.0047** |
| Swin-T　 | 原始V2　　　　　　　　| 0.6793±0.0313     | 0.6914±0.0343     | 0.6263±0.0244     | 0.6601±0.0267     | 0.6709±0.0569     | 0.5736±0.0731     | 0.9348±0.0853     |
| Swin-T　 | 新版V2（align\_proj） | 0.6646±0.0434     | **0.7015±0.0453** | 0.5942±0.0518     | 0.6264±0.0411     | 0.6325±0.1096     | 0.5499±0.1998     | 0.9723±0.0237     |
| ViT-B/16 | V1　　　　　　　　　　| **0.7104±0.0253** | **0.7222±0.0295** | 0.6289±0.0345     | **0.6686±0.0411** | 0.6969±0.0956     | 0.5490±0.1152     | 0.9512±0.0626     |
| ViT-B/16 | 原始V2　　　　　　　　| 0.6404±0.0496     | 0.6592±0.0522     | 0.5779±0.0370     | 0.6478±0.0322     | **0.7220±0.1123** | 0.4077±0.1766     | 0.9398±0.0621     |
| ViT-B/16 | 新版V2（align\_proj） | 0.6971±0.0356     | 0.7155±0.0217     | **0.6403±0.0361** | 0.6505±0.0624     | 0.6327±0.1207     | **0.6495±0.1461** | **0.9561±0.0601** |

#### 11.2 关键差值（以 AUC\_central 为主）

- **EffB0**：
  - 原始V2 vs V1：`-0.0024`
  - 新版V2 vs 原始V2：`+0.0203`
  - 新版V2 vs V1：`+0.0179`
- **Swin-T**：
  - 原始V2 vs V1：`-0.0146`
  - 新版V2 vs 原始V2：`-0.0147`
  - 新版V2 vs V1：`-0.0293`
- **ViT-B/16**：
  - 原始V2 vs V1：`-0.0700`
  - 新版V2 vs 原始V2：`+0.0567`
  - 新版V2 vs V1：`-0.0133`

#### 11.3 结论解读

- 新版对齐投影策略对 `ViT-B/16` **恢复效果显著**：相较原始V2，AUC\_central 明显回升，且 ACC/Spe/Compliance 同步改善。
- `EffB0` 在新版V2下 AUC\_central 也有提升，说明“先对齐再融合”对部分骨干确实有效。
- `Swin-T` 在当前新版配置下未获益（AUC\_central 继续下降），提示该骨干可能需要不同的投影维度、融合算子或训练超参。
- 因此，当前不宜得出“新版V2对所有骨干统一增益”的结论；更合理的结论是：**新版融合显著缓解了 ViT 的 V2 退化，但跨骨干增益仍具模型依赖性**。

### 12) FiLM 三骨干结果归档（新增）

实验设置：

- `radiomics_fusion_mode=film`
- `radiomics_topk=20`
- `film_hidden_dim=128`
- `film_gamma_scale=0.1`
- 其余协议与前述 V2 对比保持一致（`pretrained=True`, `freeze_backbone_epochs=3`, 5-fold test）
- `ViT-B/16` 同样使用 `max_instances=8` 以规避显存溢出

结果目录：

- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_effb0_film_k20`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_swin_t_film_k20`
- `HCP-Fusion Net/v2_mil_gated/checkpoints_v2_tri_source/ours_v2_tri_source_vit_b_16_film_k20`

结果汇总（`summary.csv` 的 `mean±std`）：

| 模型　　　　　　| AUC_central   | AUC_lnm       | ACC           | F1            | Sen           | Spe           | Compliance    |
| -----------------| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:|
| EffB0 (FiLM)　　| 0.6951±0.0354 | 0.6978±0.0323 | 0.6333±0.0475 | 0.6930±0.0322 | 0.7607±0.0473 | 0.4826±0.1088 | 0.9747±0.0303 |
| Swin-T (FiLM)　 | 0.6631±0.0399 | 0.6679±0.0326 | 0.6009±0.0542 | 0.6278±0.0678 | 0.6424±0.1717 | 0.5514±0.2129 | 0.9724±0.0440 |
| ViT-B/16 (FiLM) | 0.6868±0.0451 | 0.6991±0.0451 | 0.6428±0.0584 | 0.6706±0.0354 | 0.6663±0.0556 | 0.6149±0.1577 | 0.9837±0.0158 |

### 13) 原始方案 vs align\_proj vs FiLM：多指标对比与决策建议（新增）

#### 13.1 各骨干在三种融合策略下的 AUC\_central（V2）

| 模型　　 | 原始V2（legacy） | 新版V2（align\_proj） | 新版V2（FiLM） |
| ----------| -----------------:| ----------------------:| ---------------:|
| EffB0　　| 0.6708　　　　　 | 0.6911　　　　　　　　| **0.6951**　　 |
| Swin-T　 | **0.6793**　　　 | 0.6646　　　　　　　　| 0.6631　　　　 |
| ViT-B/16 | 0.6404　　　　　 | **0.6971**　　　　　　| 0.6868　　　　 |

#### 13.2 关键观察（多指标）

- **EffB0**：FiLM 在 AUC\_central、F1、Sen、Compliance 上整体最优；但 Spe 较 align\_proj 略低，属于“更偏召回”的改进。
- **Swin-T**：两种新方案均未超过原始V2 的 AUC\_central；FiLM 与 align\_proj 的表现接近，说明该骨干对融合改动更敏感，可能需要单独调参。
- **ViT-B/16**：align\_proj 的 AUC\_central 最高，FiLM 次之；但 FiLM 的 Compliance 明显更高（并带来更好的 Spe），体现“稳定性/一致性收益”。

#### 13.3 决策建议（当前阶段）

1. **若主目标是 AUC\_central 最大化**  
   - 首选：`ViT-B/16 + align_proj`  
   - 备选：`EffB0 + FiLM`

2. **若主目标是临床一致性与鲁棒阈值指标（Spe/Compliance）**  
   - 首选：`ViT-B/16 + FiLM`（高 Compliance + 较高 Spe）  
   - 备选：`Swin-T + 原始V2`（该骨干在当前实验中最稳）

3. **若希望“一套策略跨骨干泛化”**  
   - 目前不建议强行统一单一融合器；更适合“按骨干选策略”：
     - `EffB0 -> FiLM`
     - `ViT-B/16 -> align_proj`（若更重一致性可切 FiLM）
     - `Swin-T -> 先保留原始V2，再小范围调参验证`

### 14) 统一汇总：V1 + 三个版本 V2（全指标 + 关键差值）

为避免章节分散阅读，本节把 `11.1`、`12`、`13` 的核心结果合并在一起。  
其中三个版本 V2 定义如下：

- `V2-legacy`：原始方案（旧版）
- `V2-align_proj`：特征空间对齐投影
- `V2-FiLM`：FiLM 特征调制

#### 14.1 EffB0：V1 与三个 V2 全指标

| 版本 | AUC_central | AUC_lnm | ACC | F1 | Sen | Spe | Compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.6732±0.0236 | 0.6992±0.0110 | 0.6173±0.0352 | 0.6592±0.0413 | 0.7014±0.1558 | 0.5178±0.2449 | 0.9700±0.0382 |
| V2-legacy | 0.6708±0.0544 | 0.7032±0.0388 | 0.6426±0.0393 | 0.6826±0.0364 | 0.7096±0.0555 | 0.5632±0.0684 | 0.9603±0.0311 |
| V2-align_proj | 0.6911±0.0323 | 0.7037±0.0257 | 0.6034±0.0429 | 0.6228±0.0834 | 0.6290±0.1558 | **0.5744±0.1028** | 0.9608±0.0408 |
| V2-FiLM | **0.6951±0.0354** | 0.6978±0.0323 | 0.6333±0.0475 | **0.6930±0.0322** | **0.7607±0.0473** | 0.4826±0.1088 | **0.9747±0.0303** |

#### 14.2 Swin-T：V1 与三个 V2 全指标

| 版本　　　　　| AUC_central       | AUC_lnm           | ACC               | F1                | Sen               | Spe               | Compliance        |
| ---------------| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:|
| V1　　　　　　| **0.6939±0.0267** | **0.7011±0.0203** | **0.6473±0.0253** | **0.6620±0.0667** | 0.6635±0.1617     | **0.6296±0.1890** | **0.9977±0.0047** |
| V2-legacy　　 | 0.6793±0.0313     | 0.6914±0.0343     | 0.6263±0.0244     | 0.6601±0.0267     | **0.6709±0.0569** | 0.5736±0.0731     | 0.9348±0.0853     |
| V2-align_proj | 0.6646±0.0434     | **0.7015±0.0453** | 0.5942±0.0518     | 0.6264±0.0411     | 0.6325±0.1096     | 0.5499±0.1998     | 0.9723±0.0237     |
| V2-FiLM　　　 | 0.6631±0.0399     | 0.6679±0.0326     | 0.6009±0.0542     | 0.6278±0.0678     | 0.6424±0.1717     | 0.5514±0.2129     | 0.9724±0.0440     |

#### 14.3 ViT-B/16：V1 与三个 V2 全指标

| 版本　　　　　| AUC_central       | AUC_lnm           | ACC               | F1                | Sen               | Spe               | Compliance        |
| ---------------| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:| ------------------:|
| V1　　　　　　| **0.7104±0.0253** | **0.7222±0.0295** | 0.6289±0.0345     | 0.6686±0.0411     | 0.6969±0.0956     | 0.5490±0.1152     | 0.9512±0.0626     |
| V2-legacy　　 | 0.6404±0.0496     | 0.6592±0.0522     | 0.5779±0.0370     | 0.6478±0.0322     | **0.7220±0.1123** | 0.4077±0.1766     | 0.9398±0.0621     |
| V2-align_proj | 0.6971±0.0356     | 0.7155±0.0217     | 0.6403±0.0361     | 0.6505±0.0624     | 0.6327±0.1207     | **0.6495±0.1461** | 0.9561±0.0601     |
| V2-FiLM　　　 | 0.6868±0.0451     | 0.6991±0.0451     | **0.6428±0.0584** | **0.6706±0.0354** | 0.6663±0.0556     | 0.6149±0.1577     | **0.9837±0.0158** |

#### 14.4 关键差值分析（以 AUC_central 为主）

记号：

- `Δ(legacy - V1)`
- `Δ(align_proj - legacy)`
- `Δ(FiLM - legacy)`
- `Δ(最佳V2 - V1)`

| 模型　　 | Δ(legacy−V1) | Δ(align\_proj−legacy) | Δ(FiLM−legacy) | Δ(最佳V2−V1)　　　　　|
| ----------| -------------:| ----------------------:| ---------------:| ----------------------:|
| EffB0　　| -0.0024      | +0.0203               | +0.0243        | +0.0219 (FiLM)　　　　|
| Swin-T　 | -0.0146      | -0.0147               | -0.0162        | -0.0146 (legacy)　　　|
| ViT-B/16 | -0.0700      | +0.0567               | +0.0464        | -0.0133 (align\_proj) |

#### 14.5 一页式决策建议

- **如果你优先追求 AUC_central：**
  - `EffB0 -> V2-FiLM`
  - `ViT-B/16 -> V2-align_proj`
  - `Swin-T -> 先用 V2-legacy`

- **如果你优先追求临床一致性（Compliance）和更稳健阈值表现：**
  - `ViT-B/16 -> V2-FiLM`（Compliance 提升明显）
  - `EffB0 -> V2-FiLM`（F1/Sen/Compliance 更优）
  - `Swin-T -> V1` 仍是最稳参考上界

- **如果要“同一策略跨骨干统一部署”：**
  - 当前不建议强行统一；更适合“骨干特异策略”组合：
    - `EffB0: FiLM`
    - `ViT-B/16: align_proj`（若更重一致性可切 FiLM）
    - `Swin-T: legacy`（后续再做小网格微调）

---

## 十五、原始临床回归基线（新增）

为满足论文中“临床基线应基于原始术前临床变量，而非深度模型预处理后的 `clinical_features.csv`”这一要求，新增脚本：

- `HCP-Fusion Net/clinical_raw_logistic_cv5.py`

该脚本直接读取原始 Excel：

- `data/(withLNM)data3_clinical_huizong.xlsx`

并在与深度模型完全一致的患者级 5-fold 划分：

- `data/MultiModalDataset/groups_cv5_central/fold_k/{train,val,test}.txt`

上构建传统临床回归分析基线。

### 1) 标签与特征定义

- `Image`：与数据集样本名对齐（去扩展名后匹配 group 文件）
- `HISID`：患者 ID，用于患者级聚合与防止同患者信息泄漏
- `D~Z` 列：术前临床特征候选
- `AC(阳性LN总数) > 0`：定义 `LNM=1`
- `AE(侧方LN阳性数) > 0`：定义 `LateralLNM=1`
- `AG(中央组LN阳性数) > 0`：定义 `CentralLNM=1`

### 2) 建模流程

- 同一患者多张图像先聚合到患者级：
  - 连续变量取患者内中位数
  - 分类型变量取众数
- 仅在训练集内拟合预处理：
  - 连续变量中位数填补
  - 分类型变量众数填补 + one-hot 编码
- 对每个任务、每个 fold 执行：
  - 单因素 Logistic 回归筛选
  - 多因素 Logistic 回归建模
  - 导出 OR / 95\%CI / p 值
  - 在 test 集汇报 AUC / ACC / F1 / Sen / Spe

### 3) 输出目录

当前结果已保存到：

- `HCP-Fusion Net/checkpoints_raw_clinical/raw_clinical_logistic_cv5`

主要文件包括：

- `summary_mean_std.csv`
- `all_metrics.csv`
- `univariate_LNM.csv` / `univariate_CentralLNM.csv` / `univariate_LateralLNM.csv`
- `multivariate_LNM.csv` / `multivariate_CentralLNM.csv` / `multivariate_LateralLNM.csv`
- `roc_LNM.png` / `roc_CentralLNM.png` / `roc_LateralLNM.png`
- `patient_feature_audit_summary.csv`

### 4) 当前结果（5-fold test）

| 任务　　　 | AUC Mean±Std  | ACC    | F1     | Sen    | Spe    |
| ------------| --------------:| -------:| -------:| -------:| -------:|
| LNM　　　　| 0.7023±0.0448 | 0.6356 | 0.6705 | 0.6721 | 0.5880 |
| CentralLNM | 0.6859±0.0658 | 0.6332 | 0.6661 | 0.6712 | 0.5881 |
| LateralLNM | 0.8409±0.0902 | 0.9094 | 0.3055 | 0.2818 | 0.9720 |

### 5) 结果解读

- `CentralLNM` 的原始临床回归基线 AUC 约为 `0.686`，与先前使用预处理临床特征得到的 clinical-only 基线量级接近，说明原始临床变量本身已具备较强判别力。
- `LNM` 任务上原始临床基线达到约 `0.702` 的 AUC，可作为后续多任务深度模型的重要对照。
- `LateralLNM` 虽然 AUC 较高，但灵敏度明显偏低、特异度极高，提示其更像“保守判阳”的回归基线，后续解释时不宜只看 AUC。

### 6) 单因素与多因素筛选结果（你可以直接写进论文）

#### 6.1 单因素（p<0.05）跨 fold 最稳定特征

- `LNM`
  - `超声探及淋巴结异常（1=中央VI区）`：`5/5` fold 显著
  - `calcification(micro)`：`5/5`
  - `年龄`：`5/5`（负向）
  - `size（横径）cm`：`5/5`
  - `size（前后径）cm` / `size（长径）cm`：`4/5`
- `CentralLNM`
  - `calcification(micro)`：`5/5`
  - `年龄`：`5/5`（负向）
  - `超声探及淋巴结异常（1=中央VI区）`：`4/5`
  - `size（长/横/前后径）cm`：`4/5`
  - `性别（男）`：`4/5`
- `LateralLNM`
  - `超声探及淋巴结异常（1=中央VI区）`：`5/5`
  - `超声探及淋巴结异常（2=侧方）`：`5/5`
  - `size（前后径）cm`：`3/5`
  - `性别（男）`：`3/5`
  - `多灶（多灶=1）`：`2/5`

#### 6.2 多因素模型跨 fold 最稳定入选特征（附中位 OR）

- `LNM`
  - `超声探及淋巴结异常（1=中央VI区）`：`5/5`，中位 `OR≈6.41`
  - `calcification(micro)`：`5/5`，中位 `OR≈2.05`
  - `年龄`：`5/5`，中位 `OR≈0.95`
  - `性别（男）`：`5/5`，中位 `OR≈1.75`
  - `size（长径）cm`：`5/5`，中位 `OR≈2.03`
  - `size（前后径）cm`：`4/5`，中位 `OR≈1.87`
- `CentralLNM`
  - `超声探及淋巴结异常（1=中央VI区）`：`5/5`，中位 `OR≈4.03`
  - `calcification(micro)`：`5/5`，中位 `OR≈2.09`
  - `年龄`：`5/5`，中位 `OR≈0.96`
  - `性别（男）`：`4/5`，中位 `OR≈1.67`
  - `size（长径）cm`：`4/5`，中位 `OR≈2.74`
  - `size（前后径）cm`：`4/5`，中位 `OR≈2.12`
- `LateralLNM`
  - `超声探及淋巴结异常（2=侧方）`：`5/5`，中位 `OR≈46.19`
  - `超声探及淋巴结异常（1=中央VI区）`：`5/5`，中位 `OR≈17.45`
  - `性别（男）`：`4/5`，中位 `OR≈2.61`
  - `size（前后径）cm`：`3/5`，中位 `OR≈7.88`
  - `多灶（多灶=1）`：`3/5`，中位 `OR≈2.97`
  - `年龄`：`3/5`，中位 `OR≈0.97`

> 说明：这里 OR 是按 5 折多因素系数取中位数后换算得到；详细逐 fold 的 `OR/95%CI/p` 请直接看：
> - `checkpoints_raw_clinical/raw_clinical_logistic_cv5/univariate_*.csv`
> - `checkpoints_raw_clinical/raw_clinical_logistic_cv5/multivariate_*.csv`

---

## 十六、临床期刊重组稿（新增）

为避免现有 `overleaf_package` 和 `overleaf_package_mil` 同时混合“实验报告叙事”和“投稿论文叙事”，现已新增一个专门用于**临床期刊主文重组**的新目录：

- `overleaf_package_clinical_journal/`

这个目录的目标不是继续堆实验过程，而是把现有结果按**临床论文标准逻辑**重新组织成一版主文 + 补充材料：

- 主文：`overleaf_package_clinical_journal/main.tex`
- 主文摘要：`overleaf_package_clinical_journal/sections/abstract.tex`
- 主文引言：`overleaf_package_clinical_journal/sections/introduction.tex`
- 主文方法：`overleaf_package_clinical_journal/sections/methods.tex`
- 主文结果：`overleaf_package_clinical_journal/sections/results.tex`
- 主文讨论：`overleaf_package_clinical_journal/sections/discussion.tex`
- 主文结论：`overleaf_package_clinical_journal/sections/conclusion.tex`
- 补充材料：`overleaf_package_clinical_journal/supplement.tex`

### 1) 新主文采用的固定叙事顺序

主线不再按“你先做了哪些实验”来写，而是按期刊审稿人更容易接受的顺序来写：

1. 原始临床基线模型  
2. 传统影像组学方法  
3. 多模态深度学习融合主模型  
4. 多架构公平比较  
5. 关键模块消融验证  
6. 多源融合扩展验证  
7. 讨论、局限与结论

其中：

- **主终点固定为 `CentralLNM`**
- **统一强调患者级 `5-fold` 公平比较**
- **主比较指标统一为 `AUC mean±std`，并辅以 `ACC/F1/Sen/Spe`**

### 2) 哪些内容进入主文，哪些移入补充材料

主文保留：

- 原始临床基线的 `CentralLNM` 结果与关键变量
- 传统影像组学三条路径的汇总表
- `Attention-MIL + Clinical-first Gated + Prior` 主模型
- 8 种骨干网络的精简公平比较
- 六组消融汇总表
- 三源融合的精简主表

补充材料保留：

- 逐 fold 明细表
- V1 与多个 V2 融合版本的大表
- 原始临床的三任务完整结果

不再作为最终主文核心内容的材料：

- 早期 `A/B/C` 排查
- 日志可信度修复过程
- 路线图、完整性核对表、开发过程性验证

### 3) 使用说明

如果你要继续改临床期刊主文，优先编辑：

- `overleaf_package_clinical_journal/main.tex`

如果你要补充逐 fold 明细、方法敏感性分析、多版本三源融合表格，优先编辑：

- `overleaf_package_clinical_journal/supplement.tex`

编译说明见：

- `overleaf_package_clinical_journal/README_OVERLEAF.txt`

### 4) 当前阶段的建议

- 如果下一步目标是**先把稿子投出去**，就继续在 `overleaf_package_clinical_journal/` 上精修语言、图表和参考文献。
- 如果下一步目标是**继续做实验增强结果**，建议优先围绕三源融合中“组学特征与深度特征如何更稳定交互”继续推进，而不是再扩展更多过程性对照。

---

## 十七、投稿升级策略（新增）

如果你当前最关心的问题是：

- 为什么多种影像/组学/深度融合方法没有明显超过原始临床基线？
- 下一步该继续做什么，才能更像高质量期刊而不是继续堆实验？

请优先查看：

- `PUBLICATION_UPGRADE_STRATEGY.md`

这份文档已经把当前阶段的判断整理成可直接执行的路线，核心结论是：

- `CentralLNM` 任务本身确实难，原始临床基线已经较强；
- 当前影像分支更像提供“有限但稳定的增量信息”，而不是替代临床；
- 下一阶段不应继续停留在普通拼接融合，而应重点推进：
  - 层级结构建模（`CentralLNM ⊆ LNM`）
  - 患者级弱监督可解释 MIL
  - 个体化证据融合（clinical / deep / radiomics）
  - 外部验证、统计检验、校准与净获益分析

如果你要继续做高质量投稿准备，建议执行顺序为：

1. 先把论文主问题改成“影像是否带来稳定增量价值”
2. 再做一个真正可写成创新点的方法
3. 最后补齐高质量期刊最看重的临床证据

---

## 十八、A+C 结构升级实现（新增）

为了把 `PUBLICATION_UPGRADE_STRATEGY.md` 中最值得优先推进的路线真正落到代码里，当前已在：

- `HCP-Fusion Net/v2_mil_gated/`

中新增两类核心能力：

### 1) 路线 A：层级结构建模

新增参数：

- `--head_mode legacy|hierarchical`

其中：

- `legacy`：沿用原双头输出
- `hierarchical`：先预测 `LNM`，再预测 `CentralLNM | LNM`，最终组合得到 `P(CentralLNM)`

当前实现文件：

- `HCP-Fusion Net/v2_mil_gated/networks.py`
- `HCP-Fusion Net/v2_mil_gated/loss.py`

### 2) 路线 C：个体化证据融合

新增参数：

- `--fusion_mode legacy|evidence_moe`

其中：

- `legacy`：沿用原来的临床主导门控/拼接融合
- `evidence_moe`：显式区分 `clinical / deep / radiomics` 三路证据，并学习患者级动态权重

同时新增相关超参数：

- `--evidence_hidden_dim`
- `--evidence_entropy_alpha`
- `--evidence_temperature`
- `--conditional_loss_alpha`

### 3) 当前兼容性说明

当前设计遵循“兼容扩展”原则：

- 旧模式仍可直接运行
- 新模式是通过额外开关启用，不会破坏已有 `legacy / align_proj / FiLM` 结果目录
- `summary.json/csv`、`predictions.npz`、ROC 绘图链路仍然保留

### 4) 最小运行示例

#### 4.1 旧模式 smoke

```bash
cd /data2/wjb/code/PTMC_LNM
/data2/wjb/anaconda3/envs/pumc/bin/python "HCP-Fusion Net/v2_mil_gated/train.py" \
  --data_root "/data2/wjb/code/PTMC_LNM/data/MultiModalDataset" \
  --groups_root groups_cv5_central \
  --save_dir "HCP-Fusion Net/v2_mil_gated/checkpoints_v2_ac_smoke" \
  --exp_name legacy_smoke_resnet18 \
  --model_mode attention_mil_gated \
  --backbone resnet18 \
  --pretrained False \
  --epochs 1 \
  --folds 1 \
  --gpu_id 0 \
  --num_workers 0
```

#### 4.2 A+C 新模式 smoke

```bash
cd /data2/wjb/code/PTMC_LNM
/data2/wjb/anaconda3/envs/pumc/bin/python "HCP-Fusion Net/v2_mil_gated/train.py" \
  --data_root "/data2/wjb/code/PTMC_LNM/data/MultiModalDataset" \
  --groups_root groups_cv5_central \
  --save_dir "HCP-Fusion Net/v2_mil_gated/checkpoints_v2_ac_smoke" \
  --exp_name ac_smoke_resnet18 \
  --model_mode attention_mil_gated \
  --backbone resnet18 \
  --pretrained False \
  --epochs 1 \
  --folds 1 \
  --gpu_id 0 \
  --num_workers 0 \
  --use_radiomics True \
  --radiomics_root "/data2/wjb/code/PTMC_LNM/HCP-Fusion Net/checkpoints_roi_radiomics/central_roi_radiomics_cv5" \
  --head_mode hierarchical \
  --fusion_mode evidence_moe
```

### 5) 三源实验入口

`run_tri_source_experiments.py` 现在还新增了：

- `--run_ac`
- `--head_mode`
- `--fusion_mode`

可用于在原有 `V1 / V2` 之外，额外拉起一条 A+C 新结构实验。
