# PTMC CentralLNM 投稿升级策略

## 1. 当前阶段的核心判断

当前结果更像“两部分叠加”：

- 任务本身确实难：`CentralLNM` 的术前信号有限，且原始临床基线已经较强。
- 方法仍未充分发挥：影像与组学并未被组织成真正能够证明“临床增量价值”的结构。

在统一患者级 5-fold 协议下，最关键的数字是：

- 原始临床 Logistic：`0.6859 ± 0.0658`
- ROI Radiomics-only：`0.5756 ± 0.0538`
- Clinical+ROI Radiomics：`0.6923 ± 0.0696`
- Deep+Clinical（ViT-B/16）：`0.7104 ± 0.0253`
- Tri-Fusion（ViT-B/16）：`0.6404 ± 0.0496`

因此，当前最合理的表述不是“影像没有价值”，而是：

- 影像没有形成足够强的替代价值
- 影像已经显示出有限但更稳定的增量价值
- 多源融合当前暴露的是融合机制问题，而不是“加组学必然无效”

## 2. 为什么当前方法没有大幅超过原始临床

### 2.1 更像任务上限的因素

- 原始临床基线已经不低，说明许多可解释信号已被术前变量捕获。
- `CentralLNM` 的患者级标签由切面集合弱监督聚合得到，天然存在噪声。
- 当前全部主结果仍来自单中心内部 5-fold，真实泛化上限可能更低。

### 2.2 更像方法短板的因素

- 当前主模型本质仍是 `clinical-first gated residual fusion`，临床分支太强时，影像分支容易只剩边际残差。
- ROI radiomics 与深度特征的融合多数仍停留在拼接/浅层调制，难以形成互补证据。
- MIL 当前更接近标准 attention pooling，而不是“带关键证据定位的患者级弱监督”。
- `CentralLNM ⊆ LNM` 目前主要依赖 soft penalty，而不是结构化输出。

## 3. 高质量期刊更值得讲的问题

如果目标是高质量临床 AI 期刊，论文主问题建议从：

- “谁的 AUC 更高”

升级为：

- 影像是否在原始临床基线之上提供稳定增量
- 影像在哪些患者上改变了原始临床决策
- 模型是否改善了稳定性、校准性与净获益，而不只是点估计 AUC

这个转换会直接改变论文的说服力。

## 4. 方法创新路线

## 4.1 路线 A：层级结构建模

目标：把 `CentralLNM ⊆ LNM` 从软约束升级为结构化建模。

建议方向：

- 层级输出头：先预测 `LNM`，再在条件空间中预测 `CentralLNM`
- 显式条件概率分解：例如 `P(CentralLNM) = P(LNM) * P(CentralLNM | LNM)`
- 结构一致性损失替代单纯 hinge penalty

建议优先修改位置：

- `HCP-Fusion Net/v2_mil_gated/networks.py`
- `HCP-Fusion Net/v2_mil_gated/loss.py`
- `HCP-Fusion Net/v2_mil_gated/train.py`

预期价值：

- 更符合临床逻辑
- 更容易写成方法学创新
- 更便于解释“为什么一致性提升”

## 4.2 路线 B：患者级弱监督可解释 MIL

目标：把当前 bag-level attention 升级为“关键证据切面发现 + 患者级决策”。

建议方向：

- 关键切面挖掘
- 伪标签或一致性自训练
- 切面间对比学习
- 融合病灶 ROI 或可疑淋巴结区域先验

建议优先修改位置：

- `HCP-Fusion Net/v2_mil_gated/dataset.py`
- `HCP-Fusion Net/v2_mil_gated/networks.py`
- `HCP-Fusion Net/v2_mil_gated/make_explainability_panels_v2.py`

预期价值：

- 提高影像分支的真正信息密度
- 增强可解释性
- 更适合临床期刊对“证据切面”的叙事

## 4.3 路线 C：个体化证据融合

目标：不要再把临床、深度影像、组学简单拼接，而是学习“某个患者该信哪一路”。

建议方向：

- mixture-of-experts
- uncertainty-aware gating
- evidence-level fusion
- clinical-first but patient-specific weighting

建议优先修改位置：

- `HCP-Fusion Net/v2_mil_gated/networks.py`
- `HCP-Fusion Net/v2_mil_gated/options.py`
- `HCP-Fusion Net/v2_mil_gated/run_tri_source_experiments.py`

预期价值：

- 更适合解释为何 `V2` 没超过 `V1`
- 更可能把失败的三源融合转成真正的创新点

## 5. 哪些只是工程优化，哪些更适合投稿创新

### 5.1 常规工程优化

- 再换 backbone
- 常规调参
- 继续堆更多 concat / FiLM / 对齐变体
- 只追求 AUC 增加 `0.01`

这些重要，但更像“必要优化”，不够支撑高质量投稿主创新。

### 5.2 更适合作为论文创新点

- 层级标签结构化建模
- 患者级弱监督可解释 MIL
- 个体化证据融合
- 临床增量价值证明框架

## 6. 高质量期刊最看重的证据补强

以下内容不一定让 AUC 明显更高，但会显著提升稿件档次：

- 外部验证或时间切分验证
- bootstrap 置信区间
- DeLong 或增量检验
- 校准强化：Brier、ECE、recalibration
- DCA 与阈值区间分析
- 错误病例分析
- 临床边界病例亚组分析

## 7. 推荐执行顺序

### 第一步：先把主叙事改成“增量价值”

- 原始临床是强基线
- 影像是补充，而不是替代
- 重点回答“在哪些患者上有帮助”

### 第二步：优先推进一个真正的方法创新

优先级建议：

1. 层级结构建模
2. 个体化证据融合
3. 弱监督可解释 MIL

### 第三步：补临床发表证据

- 外部验证
- 统计检验
- 校准与净获益
- 亚组与错误分析

## 8. 当前最现实的投稿定位

现阶段最适合的高质量投稿逻辑不是：

- “我们提出了一个更复杂的融合网络”

而是：

- 原始临床基线已经很强
- 影像提供的是有限但稳定的增量信息
- 结构化与临床一致性的设计是增量能够稳定出现的关键
- 多源融合当前暴露了瓶颈，并指向下一代个体化证据融合框架

这条线更符合高质量临床 AI 期刊的审稿期待。
