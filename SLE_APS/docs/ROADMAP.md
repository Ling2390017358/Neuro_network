markdown
# SLE-APS Prediction Model Study — Project Roadmap
# SLE继发抗磷脂综合征多维度预测模型研究

## 🎯 项目目标
基于9,420例SLE患者纵向队列，构建符合TRIPOD+AI规范的APS预测体系，
产出1-2篇高水平SCI论文（目标：Annals of the Rheumatic Diseases / Arthritis & Rheumatology / Lupus Science & Medicine级别）

## 📊 数据资产
- 数据源：SLEmatrix_merged.csv
- 患者数：9,420例 | 就诊记录：79,655条
- 时间跨度：2008-08-17 至 2025-02-14
- 特征维度：254列（97定量 + 88定性 + 69其他）
- 核心输出目录：/home/ubuntu/projects/SLE_APS/analysis/output/

## 🔬 研究分为两篇论文
### Paper 1: 横断面预测模型（Cross-sectional Prediction Model）
  - 目标期刊：Arthritis & Rheumatology / Lupus Science & Medicine
  - 核心内容：基于常规实验室指标的APS风险预测模型开发与内部验证
  
### Paper 2: 纵向时序预测与早期预警（Longitudinal Prediction & EWS）
  - 目标期刊：Annals of the Rheumatic Diseases / RMD Open
  - 核心内容：纵向轨迹特征 + 深度学习时序预测 + 临床评分卡

## 📅 项目阶段 (8个Phase)

### Phase 0: 数据治理与质量控制 [优先级：🔴 Critical]
- 状态：待执行
- 目标：建立可靠的分析基础
- 关键产出：数据质量报告、清洗后数据集

### Phase 1: APS表型精确定义与队列构建 [优先级：🔴 Critical]  
- 状态：需优化（现有定义需对标2023 ACR/EULAR标准）
- 目标：构建明确定义的分析队列
- 关键产出：队列流程图（CONSORT-style）、表型定义文档

### Phase 2: 基线特征描述与单因素分析 [优先级：🟡 High]
- 状态：部分完成，需补充
- 目标：产出Table 1 + 单因素分析森林图
- 关键产出：标准化基线特征表、标准化均数差(SMD)分析

### Phase 3: 多因素预测模型开发 [优先级：🟡 High]
- 状态：部分完成，需大幅优化
- 目标：符合TRIPOD+AI规范的预测模型
- 关键产出：LASSO/RF/XGBoost/LightGBM模型 + SHAP解释

### Phase 4: 模型验证与临床效用评估 [优先级：🟡 High]
- 状态：待执行
- 目标：内部验证 + 校准 + 临床决策分析
- 关键产出：ROC/校准曲线/DCA/Net Reclassification

### Phase 5: 纵向轨迹建模 [优先级：🟢 Medium]
- 状态：部分完成，需深化
- 目标：混合效应模型 + 联合模型 + 轨迹可视化
- 关键产出：补体/凝血指标轨迹图、LCMM潜类分析

### Phase 6: 深度学习时序预测 [优先级：🟢 Medium]
- 状态：部分完成，需补充可解释性
- 目标：Transformer/LSTM + 注意力可视化 + SHAP
- 关键产出：时序预测模型 + 注意力权重热图

### Phase 7: 临床评分卡与综合报告 [优先级：🟢 Medium]
- 状态：部分完成，需优化
- 目标：床旁可用EWS评分 + Nomogram + 论文级图表
- 关键产出：Nomogram + 评分卡 + 论文全套Figure/Table

### Phase 8: 论文撰写与投稿准备 [优先级：🔵 Final]
- 状态：待执行
- 目标：产出完整论文草稿 + 补充材料
- 关键产出：Main manuscript + Supplementary + Cover letter

## 📁 项目目录结构
/home/ubuntu/projects/SLE_APS/
├── data/
│ ├── raw/ # 原始数据
│ ├── processed/ # 清洗后数据
│ └── imputed/ # 多重插补数据集
├── analysis/
│ ├── scripts/ # 分析脚本（按Phase编号）
│ │ ├── phase0_data_quality.py
│ │ ├── phase1_cohort_definition.py
│ │ ├── phase2_descriptive.py
│ │ ├── phase3_modeling.py
│ │ ├── phase4_validation.py
│ │ ├── phase5_longitudinal.py
│ │ ├── phase6_deep_learning.py
│ │ ├── phase7_ews_nomogram.py
│ │ └── phase8_manuscript_figures.py
│ ├── output/ # 分析结果
│ ├── figures/ # 论文图表
│ │ ├── main/ # 主图（6-8张）
│ │ └── supplementary/ # 补充图
│ └── tables/ # 论文表格
│ ├── main/ # 主表（3-4个）
│ └── supplementary/ # 补充表
├── models/ # 训练好的模型
├── docs/ # 文档
│ ├── ROADMAP.md
│ ├── TASKS.md
│ └── PROGRESS.md
└── manuscript/ # 论文稿件
├── paper1/
└── paper2/

markdown

## 🔧 技术栈
- Python 3.10+ | pandas | numpy | scipy
- scikit-learn | XGBoost | LightGBM
- PyTorch (Bi-LSTM, Transformer)
- SHAP | LIME | InterpretML
- lifelines (生存分析) | statsmodels (混合效应)
- matplotlib | seaborn | plotly
- tableone (Table 1生成)
- MICE (多重插补)

## ⚠️ 关键风险与缓解措施
| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| APL检测覆盖率低(14.7%) | 选择偏倚 | 多重插补 + 敏感性分析 + 逆概率加权(IPW) |
| 类别不平衡(11.5% APS) | 模型偏向多数类 | SMOTE + 下采样 + 代价敏感学习 |
| 单中心数据 | 外推性受限 | 地理时间拆分验证 + 明确声明局限性 |
| 过拟合风险 | 虚高性能 | 嵌套交叉验证 + Bootstrap内部验证 |