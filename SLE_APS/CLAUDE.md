# CLAUDE.md — SLE-APS 预测模型研究项目

> 本文件是 Claude Code 的项目上下文核心配置文件。
> 每次启动新会话时，请首先读取本文件，再读取 `docs/PROGRESS.md` 确认当前进度。

---

## 🧠 项目一句话描述

基于9,420例SLE患者（79,655条就诊记录）的纵向真实世界队列，构建符合
TRIPOD+AI规范的APS预测体系，产出2篇高水平SCI论文。

---

## 📁 项目根目录

```
/home/ubuntu/projects/SLE_APS/
```

---

## 📂 目录结构

```
/home/ubuntu/projects/SLE_APS/
├── data/
│   ├── raw/
│   │   └── SLEmatrix_merged.csv          # ⚠️ 原始数据，禁止直接修改
│   ├── processed/
│   │   ├── train_data.csv                # 训练集（2008-2020）
│   │   └── temporal_test_data.csv        # 时间验证集（2021-2025）
│   └── imputed/
│       └── imputed_data_m{1-5}.csv       # MICE多重插补数据集（m=5）
├── analysis/
│   ├── scripts/                          # 按Phase编号的分析脚本
│   │   ├── phase0_data_quality.py
│   │   ├── phase1_cohort_definition.py
│   │   ├── phase2_descriptive.py
│   │   ├── phase3_modeling.py
│   │   ├── phase4_validation.py
│   │   ├── phase5_longitudinal.py
│   │   ├── phase6_deep_learning.py
│   │   ├── phase7_ews_nomogram.py
│   │   └── phase8_manuscript_figures.py
│   ├── output/                           # 所有分析产出（CSV/模型文件）
│   ├── figures/
│   │   ├── main/                         # 投稿主图（PDF + TIFF，≥300DPI）
│   │   └── supplementary/               # 补充图
│   └── tables/
│       ├── main/                         # 主表（Word + CSV双格式）
│       └── supplementary/
├── models/                               # 训练好的模型权重
│   ├── bilstm_best.pt
│   └── transformer_best.pt
├── docs/
│   ├── ROADMAP.md                        # 项目总体规划（只读参考）
│   ├── TASKS.md                          # 任务详情与验收标准（执行依据）
│   └── PROGRESS.md                       # 实时进度追踪（每次任务后更新）
└── manuscript/
    ├── paper1/                           # 横断面预测模型论文
    └── paper2/                           # 纵向时序预测论文
```

---

## 🗃️ 核心数据资产

### 原始数据
| 属性 | 值 |
|------|----|
| 文件 | `data/raw/SLEmatrix_merged.csv` |
| 患者数 | 9,420例 |
| 就诊记录 | 79,655条 |
| 特征维度 | 254列（97定量 + 88定性 + 69其他） |
| 时间跨度 | 2008-08-17 至 2025-02-14 |
| 关键标识列 | `patientSN`（患者ID）、`visit_date`（就诊日期） |

### 数据库维度速查

| 维度 | 列数 | 代表性指标 |
|------|------|-----------|
| ID/日期 | 5 | patientSN, visit_date |
| ACR标准 | 29 | 蝶形红斑、关节炎、肾损害 |
| SLEDAI评分 | 25 | SLEDAI总分及各组分 |
| **APL抗体** | **12** | LAC, ACL-IgG/IgM/IgA, B2GP1-IgG/IgM |
| 治疗用药 | 6 | 激素剂量、HCQ、免疫抑制剂 |
| 生命体征 | 3 | 身高、体重、BMI |
| **凝血功能** | **8** | PT, APTT, TT, PTA, INR, Fbg |
| 血常规 | 9 | PLT, Hb, WBC, RBC |
| 免疫细胞流式 | 16 | CD3/4/8/19/14, NK, HLA-DR |
| **补体** | **8** | C3, C4（定量+定性） |
| 生化 | 34 | ALB, CK, AST, CysC, UA, 血脂 |
| 细胞因子 | 22 | IL-2/4/6/10/17, TNF-α |
| 尿液/肾功能 | 50 | 尿蛋白定量、尿沉渣 |
| 脑脊液 | 4 | CSF细胞计数、寡克隆区带 |
| EBV/感染 | 15 | EBV DNA, CMV, PCT |
| 甲状腺 | 7 | FT3/FT4, TSH |
| 自身抗体 | 25 | ANA, dsDNA, Sm, SSA/SSB |

> **⚠️ 重要提示**：APL抗体系列的检测覆盖率极低（ACL-IgG仅14.7%，1,382/9,420），
> 所有使用APL指标的分析必须注明选择偏倚，并提供多重插补敏感性分析。

---

## 🎯 研究设计核心参数

### APS表型定义（三种，主用定义A）

| 定义 | 名称 | 说明 | 预计例数 |
|------|------|------|---------|
| **定义A** | 2023 ACR/EULAR标准 | 临床域≥3分 + 实验室域≥3分 | 待确认（T1.1）|
| 定义B | Sydney 2006标准 | 敏感性分析用 | 待确认（T1.1）|
| 定义C | 任意APL阳性（宽泛） | 已用于初步分析 | 1,080例（11.5%）|

> **当前状态**：所有已完成的分析使用定义C（宽泛定义）。
> **必须**在T1.1完成后，用定义A重跑所有核心分析。

### 数据分割策略

| 集合 | 时间范围 | 用途 |
|------|---------|------|
| 训练集 | 2008-2020 | 模型开发 + 10折交叉验证 |
| 时间验证集 | 2021-2025 | 独立内部验证（时间外推） |
| Bootstrap | B=1000 | Optimism校正，在训练集上执行 |

### 类别不平衡处理规则（必须遵守）

```
⚠️ SMOTE / 过采样 / 下采样 只能在训练折内执行，
   严禁在整个数据集上执行后再分割，避免数据泄露！
```

处理方案（按优先级）：
1. `class_weight='balanced'`（代价敏感，所有模型默认开启）
2. SMOTE（仅在内层训练折内，管道内执行）
3. 下采样（仅作为敏感性分析对比）

---

## 📊 已确认的关键结果（禁止覆盖）

以下结果已产出，新分析不得使数字产生矛盾，如有差异需在PROGRESS.md记录原因：

### 单因素分析（Phase 2）
- FDR校正后 **28/34** 个生物标志物 p_adj < 0.05
- Top 3：SLEDAI总分（+120%，p=1.10×10⁻¹³⁶）、就诊次数（+125%）、APTT（+7.0%）
- LAC阳性率：APS组55.6% vs 对照组0%（OR=∞）

### 机器学习模型（Phase 3，10折嵌套CV，排除诊断标准特征后）
| 模型 | AUC |
|------|-----|
| Gradient Boosting | **0.8998 ± 0.0162** |
| XGBoost | 0.8979 ± 0.0130 |
| Stacking | 0.8945 ± 0.0134 |
| LightGBM | 0.8935 ± 0.0112 |
| 随机森林 | 0.8934 ± 0.0138 |
| LASSO逻辑回归 | 0.7361 ± 0.0407 |

### Phase 4 验证结果
- Bootstrap乐观校正AUC: 0.9136 (95%CI: 0.9014–0.9274)
- 随机分层holdout验证AUC: 0.9195
- 时间验证AUC (敏感性分析): 0.5439 (人口偏移)
- NRI: 1.635 (RF vs LR), IDI: 0.125

### 纵向轨迹（Phase 5）
- C3斜率差：APS +0.0352 vs 对照 +0.0271（p=9.27×10⁻⁶）
- APTT斜率差：APS -1.174 vs 对照 -0.775 s/年（p=1.72×10⁻³）

### 深度学习（Phase 6，6,314例有效时序患者）
| 模型 | 测试AUC |
|------|---------|
| Bi-LSTM | 0.8002 |
| Transformer | **0.8099** |

### EWS评分（Phase 7，v1.0，9变量，满分27分）
- AUC = 0.7118，最佳截断 = 5分，较文献模型 +0.053
- 风险分层：低危(0-4分,6.3%) → 中危(5-8分,17.7%) → 高危(9-12分,25.7%) → 极高危(≥13分,23.8%)

---

## 🔬 两篇论文目标

### Paper 1：横断面预测模型
- **目标期刊**：Arthritis & Rheumatology（优先）/ Lupus Science & Medicine
- **核心卖点**：大样本（n=9,420）+ 多模型比较 + TRIPOD+AI规范 + Nomogram
- **报告规范**：TRIPOD+AI checklist（目标 ≥90%完成）
- **字数限制**：≤4,000词正文
- **主图**：6张（流程图/APL谱系/森林图/SHAP/ROC+校准/DCA+Nomogram）

### Paper 2：纵向时序预测与早期预警
- **目标期刊**：Annals of the Rheumatic Diseases（优先）/ RMD Open
- **核心卖点**：纵向轨迹（LME+LCMM）+ Transformer + 注意力可视化 + EWS v2.0
- **报告规范**：STROBE + TRIPOD+AI
- **字数限制**：≤4,500词正文
- **主图**：6张（轨迹Panel/LCMM/KM曲线/DL架构/注意力热图/EWS v2.0）

---

## 🛠️ 技术栈与环境规范

### Python环境

```python
# 必须使用固定版本（更新requirements.txt时同步修改）
python == 3.10+
pandas, numpy, scipy
scikit-learn
xgboost, lightgbm
torch  # PyTorch，用于Bi-LSTM和Transformer
shap
lifelines          # 生存分析（KM/Cox/Fine-Gray）
statsmodels        # 混合效应模型（MixedLM）
tableone           # Table 1标准化生成
matplotlib, seaborn
```

### 随机种子规范

```python
# 所有涉及随机性的操作，统一使用：
RANDOM_SEED = 42

# 适用范围：
# - train_test_split / KFold
# - 所有sklearn estimators（random_state=42）
# - SMOTE
# - PyTorch（torch.manual_seed(42)）
# - numpy（np.random.seed(42)）
```

### 统计检验规范

| 场景 | 方法 |
|------|------|
| 两组连续变量 | Mann-Whitney U（非参数，数据多为偏态） |
| 两组分类变量 | χ²检验 / Fisher精确检验（期望频数<5时用Fisher）|
| 多重比较校正 | Benjamini-Hochberg FDR |
| AUC比较 | DeLong检验（`scipy`或`rpy2`调用`pROC`包）|
| 显著性阈值 | α=0.05（未校正）/ p_adj<0.05（FDR校正后）|

### 图表制作规范

```python
# 所有发表图表必须满足：
DPI = 300                    # 最低分辨率
FORMAT = ['pdf', 'tiff']     # 双格式输出
FONT = 'Arial'
FONT_SIZE_LABEL = 8          # 图内标注
FONT_SIZE_TITLE = 10         # 图标题
COLORMAP = 'viridis'         # 或 ColorBrewer色盲友好配色

# 禁止使用：纯红绿配色（色盲不友好）
```

---

## 🚦 执行规则（Claude Code必读）

### 任务执行前
1. **读取 `docs/PROGRESS.md`** — 确认当前阶段状态和最新产出文件
2. **检查依赖任务** — 确认所有前置任务已完成（依赖关系见TASKS.md）
3. **确认数据文件存在** — 验证所需CSV/模型文件在output/目录中
4. **声明当前执行的任务ID** — 格式：`[执行] T3.2 多模型训练与调参`

### 任务执行中
1. **严格按TASKS.md的验收标准执行** — 不得随意省略验收项
2. **APL分析必须标注选择偏倚** — 每个涉及APL的输出表格添加脚注
3. **禁止在已有结果基础上直接覆盖** — 新结果用新文件名或版本号区分
4. **代码添加必要注释** — 关键统计选择必须注释说明原因

### 任务完成后
1. **更新 `docs/PROGRESS.md`** — 将任务状态改为✅，填写完成日期和产出文件路径
2. **更新整体完成度百分比**
3. **记录任何偏差** — 若结果与预期不符，在PROGRESS.md的"会议记录"部分记录

### 遇到错误时
1. **数据维度不符**：先检查是否使用了正确的数据文件（raw vs processed）
2. **内存不足**：使用分块读取（`chunksize`参数）或降采样调试
3. **模型不收敛**：检查是否做了特征标准化，树模型通常不需要，线性模型必须
4. **APL覆盖率问题**：确认是否在`has_APL_test==True`子集上分析，并在报告中注明

---

## ⚠️ 关键风险与强制规则

### 🔴 绝对禁止

```
❌ 直接修改 data/raw/SLEmatrix_merged.csv
❌ 在整体数据集上执行SMOTE后再做train/test split（数据泄露）
❌ 用测试集/验证集数据调参（超参数必须只在训练集上调）
❌ 报告AUC时不同时报告95%CI
❌ 对APL指标分析时不注明14.7%覆盖率偏倚
❌ 覆盖已记录的关键结果数字（如RF AUC=0.818）而不记录原因
```

### 🟡 必须注意

```
⚠️ 所有时序分析只使用有≥2次就诊记录的患者（n=7,349）
⚠️ 深度学习时序数据只使用有≥2次就诊的患者（n=6,314）
⚠️ 生存分析的竞争风险：考虑死亡/失访作为竞争事件
⚠️ 混合效应模型收敛性：若不收敛，先检查随机效应结构，考虑去掉随机斜率
⚠️ 图表色盲友好：完成后用Coblis模拟检验
```

---

## 📋 当前阻塞项（新会话首先关注）

> 详细状态见 `docs/PROGRESS.md`

| 优先级 | 阻塞项 | 对应任务 | 状态 |
|--------|--------|---------|------|
| 🔴 P0 | 2023 ACR/EULAR APS标准未实施，所有分析基于宽泛定义C | T1.1 | ⏳ 待执行 |
| 🔴 P0 | ~Bootstrap内部验证未完成~ | T4.1 | ✅ 已完成 |
| 🔴 P0 | ~数据泄露问题（acr_/sledai_/acl_特征）~ | T3.2 | ✅ 已修复（V3） |
| 🟡 P1 | ~MICE多重插补数据集~ | T0.3 | ✅ 已完成 |
| 🟡 P1 | ~XGBoost/LightGBM/Stacking模型~ | T3.2 | ✅ 已完成 |
| 🟡 P1 | ~SHAP全套图表~ | T3.3 | ✅ 已完成 |
| 🟡 P1 | 时间验证集人口偏移（21.4% vs 4.86% APS率） | T4.1 | ⚠️ 需讨论 |
| 🟢 P2 | ~DL模型可解释性（注意力热图）~ | T6.4 | ✅ 已完成 |

---

## 📌 快速参考

### 常用文件路径

```bash
# 原始数据
data/raw/SLEmatrix_merged.csv

# 已产出的关键文件（已确认存在）
analysis/output/patient_level_data.csv        # 患者级聚合（9,420×184）
analysis/output/visit_level_data.csv          # 就诊级（79,655×258）
analysis/output/univariate_analysis.csv       # 单因素分析结果
analysis/output/feature_importance.csv        # 特征重要性排名
analysis/output/model_performance.csv         # Phase 3模型性能
analysis/output/survival_data.csv             # 生存分析数据
analysis/output/ews_scores.csv                # EWS v1.0评分
analysis/output/ews_risk_stratification.csv   # EWS风险分层
models/bilstm_best.pt                         # Bi-LSTM权重（AUC=0.8002）
models/transformer_best.pt                    # Transformer权重（AUC=0.8099）
```

### 三个计划文件说明

| 文件 | 用途 | 更新频率 |
|------|------|---------|
| `docs/ROADMAP.md` | 项目总体规划和阶段目标 | 仅在项目目标变更时 |
| `docs/TASKS.md` | 每个任务的详细描述和验收标准 | 仅在任务内容变更时 |
| `docs/PROGRESS.md` | 实时进度追踪 | **每次完成任务后必须更新** |

---

## 🔄 新会话启动检查清单

```
□ 1. 读取本文件（CLAUDE.md）
□ 2. 读取 docs/PROGRESS.md 确认当前进度
□ 3. 确认当前最高优先级阻塞项
□ 4. 确认要执行的任务ID及其依赖是否已完成
□ 5. 验证所需数据文件存在
□ 6. 声明本次会话目标任务
□ 7. 执行完成后更新 docs/PROGRESS.md
```

---

*项目启动：2026-04-28 | 预计产出：2篇SCI论文 | 维护：Claude Code*
```

---

## 说明

这份 `CLAUDE.md` 的设计遵循了以下原则：

### 为什么这样写

1. **新会话即可上手** — 包含数据库完整结构速查、已确认结果数字、文件路径，无需翻其他文件就能定位上下文

2. **防错机制** — "绝对禁止"和"必须注意"两级规则，专门针对这个项目的高风险操作（数据泄露、选择偏倚标注、覆盖已有结果）

3. **三表合一定位** — 明确了ROADMAP/TASKS/PROGRESS三个文件各自的职责和更新频率，Claude Code知道执行后该更新哪个文件

4. **硬编码关键结论** — 已确认的AUC/p值等数字写入文件，防止新分析产生矛盾结果时被悄悄覆盖

5. **分析规范统一** — 统计检验方法、随机种子、图表DPI等全部标准化，保证不同会话产出的一致性
