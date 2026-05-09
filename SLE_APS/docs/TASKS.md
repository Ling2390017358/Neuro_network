# SLE-APS Project — Task Breakdown  
# 每个Task包含：ID、描述、依赖、验收标准、预估复杂度  

---  

## Phase 0: 数据治理与质量控制  

### T0.1 数据加载与初始探索  
- **依赖**: 无  
- **复杂度**: ⭐⭐  
- **描述**:   
  1. 加载 SLEmatrix_merged.csv  
  2. 验证数据维度（预期：79,655行 × 254列）  
  3. 生成各列数据类型、唯一值数量、缺失率汇总表  
  4. 检查patientSN唯一性和visit_date格式  
  5. 检查异常值（IQR方法标记extreme outliers）  
- **验收标准**:   
  - 产出 `data_quality_overview.csv` 和 `data_quality_report.html`  
  - 每列缺失率可视化热图 `missing_heatmap.png`  

### T0.2 缺失数据模式分析  
- **依赖**: T0.1  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 使用 `missingno` 或自定义函数分析缺失模式（MCAR/MAR/MNAR）  
  2. 对APL抗体系列（ACL-IgG/IgM/IgA, B2GP1-IgG/IgM, LAC）进行缺失机制Little's MCAR检验  
  3. 分析APL检测与临床特征的关联（验证选择偏倚）  
  4. 按年份统计APL检测覆盖率变化趋势  
- **验收标准**:  
  - 产出缺失模式分析报告  
  - 确认APL检测的临床适应证偏倚量化  

### T0.3 多重插补与敏感性分析框架  
- **依赖**: T0.2  
- **复杂度**: ⭐⭐⭐⭐  
- **描述**:  
  1. 使用MICE (Multiple Imputation by Chained Equations) 生成m=5个插补数据集  
  2. 对定量变量使用PMM（预测均值匹配），定性变量使用逻辑回归  
  3. 建立完整病例分析(CCA) vs 多重插补(MI) vs 逆概率加权(IPW)三种策略的对比框架  
  4. 选取核心分析（如主模型AUC），验证三种策略结果一致性  
- **验收标准**:  
  - 产出5个插补数据集 `imputed_data_m{1-5}.csv`  
  - 敏感性分析结果对比表  

### T0.4 异常值处理与特征工程  
- **依赖**: T0.1  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 对定量指标进行Winsorize（1st/99th百分位截断）  
  2. 构建患者级聚合特征：  
     - 均值、中位数、最大值、最小值、标准差、变异系数  
     - 斜率（线性回归 vs 时间）  
     - 首末差值、异常值计数（超出正常参考范围次数）  
  3. 构建复合特征：  
     - APL三联阳性评分（LAC + ACL-IgG + B2GP1-IgG）  
     - 凝血异常综合指数  
     - 补体活化指数（C3↓ + C4↓复合评分）  
  4. 特征相关性筛选（Spearman相关>0.85者保留临床意义更强的一个）  
- **验收标准**:  
  - 产出 `patient_features_engineered.csv`（含所有聚合+复合特征）  
  - 特征相关性热图  

---  

## Phase 1: APS表型精确定义与队列构建  

### T1.1 2023 ACR/EULAR APS标准实施  
- **依赖**: T0.1  
- **复杂度**: ⭐⭐⭐⭐  
- **描述**:  
  1. 实现2023 ACR/EULAR APS分类标准的加权评分系统：  
     - 入选标准：至少1次APL阳性（3年内重复确认）  
     - 临床域评分（血栓事件/妊娠病态/微血管病变）  
     - 实验室域评分（aPL谱系和滴度加权）  
     - 临床域≥3分 且 实验室域≥3分 → 分类为APS  
  2. 同时保留旧Sydney标准定义，用于敏感性分析  
  3. 构建三种APS定义：  
     - 定义A：2023 ACR/EULAR（primary）  
     - 定义B：Sydney 2006（sensitivity analysis）  
     - 定义C：任何APL阳性（broad definition）  
- **验收标准**:  
  - 产出三种定义下的APS患者数量和Venn图  
  - APS分类标准评分明细表  

### T1.2 队列纳排流程图  
- **依赖**: T1.1  
- **复杂度**: ⭐⭐  
- **描述**:  
  1. 生成CONSORT-style患者筛选流程图  
  2. 记录每步排除原因和患者数量：  
     - 总数据库 → 确认SLE诊断 → 排除数据不完整 → 排除随访不足 → 最终分析集  
  3. 对比被排除者vs入组者的基线特征（确认无选择偏倚）  
- **验收标准**:  
  - 产出高质量流程图 `flow_chart.pdf/png`  
  - 排除人群特征对比表  

### T1.3 时间分割策略  
- **依赖**: T1.1  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 按照时间顺序将数据分为：  
     - 训练集：2008-2020年入组患者（约70%）  
     - 时间验证集：2021-2025年入组患者（约30%）  
  2. 同时准备10折分层交叉验证方案（在训练集上）  
  3. 记录训练集/验证集基线特征差异  
- **验收标准**:  
  - 产出 `train_data.csv` 和 `temporal_test_data.csv`  
  - 训练/验证集基线对比表  

---  

## Phase 2: 基线特征描述与单因素分析  

### T2.1 标准化Table 1生成  
- **依赖**: T1.2  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 使用 `tableone` 包生成标准化基线特征表  
  2. 连续变量：均数±标准差（正态）或中位数(IQR)（偏态），正态性用Shapiro-Wilk检验  
  3. 分类变量：频数(百分比)  
  4. 组间比较：连续变量用Mann-Whitney U / t检验，分类变量用χ²/Fisher精确检验  
  5. 计算标准化均数差(SMD)，>0.1视为不均衡  
  6. 分列：APS组 vs 对照组 vs 总体，附p值和SMD  
  7. 按临床维度分组展示：人口学、疾病活动度、凝血、补体、免疫、肾脏  
- **验收标准**:  
  - 产出可直接投稿的 `Table1.csv` 和 `Table1_formatted.docx`  
  - SMD Love plot `smd_plot.png`  

### T2.2 APL抗体谱系深度分析  
- **依赖**: T2.1  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 分析7种APL抗体（LAC, ACL-IgG/IgM/IgA, B2GP1-IgG/IgM, 合计）的：  
     - 阳性率对比（多阈值：10/20/40/80 U/mL）  
     - 滴度分布（箱线图 + 小提琴图）  
     - 抗体组合模式（单阳/双阳/三阳率）  
  2. 分析APL阳性的时间持续性（间隔≥12周确认）  
  3. 计算APL阳性积分（参照2023 ACR/EULAR实验室评分域）  
- **验收标准**:  
  - 产出APL谱系分析图表  
  - APL组合模式堆叠柱状图  

### T2.3 单因素分析与效应值可视化  
- **依赖**: T2.1  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 对所有定量指标进行单因素逻辑回归，获取粗OR及95%CI  
  2. 按临床维度分组的森林图（Forest Plot）  
  3. 火山图（Volcano Plot）展示效应大小vs统计显著性  
  4. 多重比较校正（Benjamini-Hochberg FDR）  
  5. 效应量计算：Cohen's d（连续）或OR（分类）  
- **验收标准**:  
  - 产出 `Figure_ForestPlot.pdf` 和 `Figure_VolcanoPlot.pdf`  
  - 单因素分析完整结果表 `univariate_results.csv`  

---  

## Phase 3: 多因素预测模型开发  

### T3.1 特征选择管道  
- **依赖**: T2.3, T0.4  
- **复杂度**: ⭐⭐⭐⭐  
- **描述**:  
  1. 三阶段特征选择：  
     - 阶段1：单因素筛选（p<0.1保留）  
     - 阶段2：LASSO路径选择（lambda.1se和lambda.min两条路径）  
     - 阶段3：Boruta算法（随机森林包装法）确认  
  2. 最终保留被≥2种方法选中的特征  
  3. 多重共线性检查（VIF>5剔除）  
  4. 临床先验知识辅助判断（不应剔除已知重要因子如APTT、C3）  
- **验收标准**:  
  - 特征选择Venn图  
  - 最终入模特征列表及选择理由  

### T3.2 多模型训练与调参  
- **依赖**: T3.1  
- **复杂度**: ⭐⭐⭐⭐⭐  
- **描述**:  
  1. 训练以下模型（嵌套交叉验证：外层10折 × 内层5折调参）：  
     - 逻辑回归（含L1/L2正则化）  
     - 随机森林（n_estimators, max_depth, min_samples调参）  
     - XGBoost（含learning_rate, max_depth, subsample, colsample调参）  
     - LightGBM（同上）  
     - 弹性网络逻辑回归（ElasticNet α, l1_ratio调参）  
     - 集成模型（Stacking：以上模型作base，LR作meta）  
  2. 处理类别不平衡：  
     - SMOTE过采样（仅在训练折内执行，避免数据泄露）  
     - 代价敏感学习（class_weight='balanced'）  
     - 下采样（作为敏感性分析）  
  3. 对每个模型记录：AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1, Brier Score  
- **验收标准**:  
  - 模型性能对比表 `model_comparison.csv`  
  - 嵌套CV结果箱线图  
  - 最优模型超参数记录  

### T3.3 SHAP可解释性分析  
- **依赖**: T3.2  
- **复杂度**: ⭐⭐⭐⭐  
- **描述**:  
  1. 对最优模型和所有tree-based模型执行SHAP分析：  
     - SHAP Summary Plot（beeswarm）  
     - SHAP Feature Importance Bar Plot  
     - SHAP Dependence Plot（Top 6特征，含interaction）  
     - SHAP Force Plot（选取典型病例：高风险正确/误判各2例）  
     - SHAP Waterfall Plot（个体化解释）  
  2. 按亚组的SHAP值对比（男vs女、年轻vs年长、高vs低SLEDAI）  
  3. SHAP interaction effects 矩阵  
- **验收标准**:  
  - 产出 `Figure_SHAP_Summary.pdf` 等全套SHAP图  
  - SHAP值排名表 `shap_importance.csv`  

### T3.4 分层模型构建（简约 → 完整）  
- **依赖**: T3.2  
- **复杂度**: ⭐⭐⭐  
- **描述**:  
  1. 构建递进式模型进行对比：  
     - 模型A：仅人口学+SLEDAI（3-4变量）  
     - 模型B：+凝血指标（+3变量：APTT, PT_INR, PLT）  
     - 模型C：+补体（+2变量：C3, C4）  
     - 模型D：+APL抗体（+2变量：ACL-IgG, LAC状态）  
     - 模型E：全量最优特征集  
  2. 每步增加的AUC增量（ΔC-statistic）和NRI/IDI  
  3. 使用DeLong检验比较嵌套模型AUC差异显著性  
- **验收标准**:  
  - 分层模型对比表  
  - 增量预测价值图（incremental discrimination improvement）  

---  

## Phase 4: 模型验证与临床效用评估  

### T4.1 内部验证  
- **依赖**: T3.2  
- **复杂度**: ⭐⭐⭐⭐  
- **描述**:  
  1. Bootstrap内部验证（B=1000次重抽样）  
  2. 计算optimism-corrected AUC  
  3. 校准曲线（calibration plot with Hosmer-Lemeshow + Brier score）  
  4. 校准斜率和截距  
  5. 时间验证集（2021-2025）独立验证  
- **验收标准**:  
  - 产出 `Figure_ROC_AllModels.pdf`  
  - 产出 `Figure_Calibration.pdf`  
  - Bootstrap验证结果表  

### T4.2 决策曲线分析 (DCA)
- **依赖**: T4.1
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 计算各模型在不同阈值概率（1%~50%）下的净获益 (Net Benefit)
  2. 绘制标准DCA曲线（包含"全部治疗"和"不治疗"参考线）
  3. 对主模型（随机森林/XGBoost）、简约模型（4变量）和EWS评分进行同图对比
  4. 计算临床净获益区间（医生可接受的阈值概率范围）
  5. 计算干预节省率（Interventions Avoided per 100 patients）
- **验收标准**:
  - 产出 `Figure_DCA.pdf`（发表级别）
  - 净获益数据表 `dca_results.csv`
  - 确定最优临床决策阈值并给出临床解释

### T4.3 净重分类改善指数与综合判别改善指数
- **依赖**: T4.1
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 计算连续型 NRI（Net Reclassification Improvement）
  2. 计算 IDI（Integrated Discrimination Improvement）
  3. 以模型A（基础模型）为参照，依次计算各递进模型的ΔC/NRI/IDI
  4. 使用 DeLong 检验比较嵌套模型 AUC 差异显著性（含 95%CI）
  5. 可视化增量预测价值（Step-wise AUC improvement plot）
- **验收标准**:
  - 产出增量预测价值汇总表 `incremental_value.csv`
  - 产出 `Figure_Incremental_AUC.pdf`

### T4.4 亚组分析与交互效应检验
- **依赖**: T4.1
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. 预定义亚组（各亚组需 ≥30 名 APS 患者）：
     - 性别（男/女）
     - 年龄（<35 / 35-50 / >50 岁）
     - 疾病活动度（SLEDAI <10 / ≥10）
     - 病程（<5年 / ≥5年）
     - 肾脏受累（是/否）
     - APL检测完整性（完整三联/部分/未检测）
  2. 各亚组分别计算模型 AUC（含 95%CI）
  3. 使用 Cochran's Q 检验亚组间异质性
  4. 交互效应检验（Likelihood ratio test for interaction term）
  5. 亚组 Forest Plot 展示
- **验收标准**:
  - 产出亚组分析 Forest Plot `Figure_Subgroup.pdf`
  - 亚组 AUC 汇总表及异质性检验结果

---

## Phase 5: 纵向轨迹建模

### T5.1 混合效应模型（Linear Mixed-Effects Model）
- **依赖**: T1.3, T0.4
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. 对核心生物标志物（C3、C4、APTT、PLT、Hb、CRP、SLEDAI）构建线性混合效应模型：
     - 固定效应：时间、APS状态、时间×APS交互项、协变量（年龄、性别、激素用量）
     - 随机效应：患者随机截距 + 随机斜率
  2. 使用 `statsmodels` MixedLM 或 `lme4`（R 通过 rpy2 调用）实现
  3. 计算各生物标志物在 APS 组 vs 对照组的时间斜率差异（Δslope）及其 95%CI
  4. 绘制分组轨迹图（带置信带），展示 APS 组与对照组的时间动态差异
  5. 对斜率显著性进行 FDR 多重校正
- **验收标准**:
  - 产出混合效应模型结果表（含固定效应系数、SE、p值）
  - 产出分组轨迹图 `Figure_Trajectory_Panel.pdf`（论文主图候选）
  - FDR 校正后显著差异指标列表

### T5.2 潜在类别混合模型（LCMM）
- **依赖**: T5.1
- **复杂度**: ⭐⭐⭐⭐⭐
- **描述**:
  1. 对补体（C3+C4复合轨迹）和凝血（APTT+PLT复合轨迹）分别拟合 LCMM
  2. 拟合 K=2~5 类模型，以 BIC/AIC 选择最优类别数
  3. 描述各潜在轨迹类别的：
     - 生物标志物变化模式（上升/下降/平台/双相）
     - 对应的 APS 发生率
     - 基线临床特征差异
  4. 将轨迹类别作为新特征纳入 Phase 3 预测模型（联合模型）
  5. 对主要结果使用 `hlme`（R/rpy2）或 Python 自定义实现
- **验收标准**:
  - 产出轨迹类别划分结果（每位患者的类别概率）
  - 产出 `Figure_LCMM_Trajectories.pdf`
  - 各类别的 APS 发生率对比表

### T5.3 生存分析深化
- **依赖**: T1.1, T5.1
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. 以首次确认 APS 事件为终点，构建时间-事件数据集
  2. Kaplan-Meier 生存曲线（按 APL 滴度分层：阴性/低滴度/高滴度/三联阳性）
  3. Cox 比例风险模型：
     - 单因素 Cox（逐一检验）
     - 多因素 Cox（纳入经 Phase 3 筛选的特征）
     - 比例风险假设检验（Schoenfeld 残差）
     - 时依协变量分析（时变的 SLEDAI、补体值）
  4. Fine-Gray 竞争风险模型（竞争事件：死亡或失访）
  5. 联合模型（Joint Model）：整合纵向生物标志物轨迹与生存终点
- **验收标准**:
  - 产出 `Figure_KM_APL_Stratified.pdf`（含 log-rank p值和风险人数表）
  - Cox 回归结果表（HR + 95%CI + p值）
  - Joint Model 估计结果及关联参数

### T5.4 时间依赖性 ROC 分析
- **依赖**: T5.3
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 计算关键生物标志物（APTT、C3、PLT、ACL-IgG）的时间依赖性 AUC
     - 预测时间点：随访后 1年、3年、5年的 APS 发生
  2. 使用 `timeROC` 方法（Blanche 等，2013）
  3. 与横断面模型（Phase 3）在相同时间窗的预测性能对比
  4. 确定最优预测时间窗和标志物组合
- **验收标准**:
  - 产出时间依赖性 ROC 曲线图 `Figure_TimeROC.pdf`
  - 不同随访时间点的 AUC 汇总表

---

## Phase 6: 深度学习时序预测

### T6.1 时序数据集构建与预处理
- **依赖**: T0.4, T1.3
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. 构建用于深度学习的时序数据格式：
     - 以就诊为时间步，选取 21 个核心生物标志物作为特征
     - 定义预测任务：利用前 N 次就诊预测 APS 发生（N=2,4,6,8）
     - 标签定义：在预测窗口（6/12个月）内是否确诊APS
  2. 时序缺失值处理策略：
     - 前向填充（Last Observation Carried Forward, LOCF）
     - 时间间隔编码（ΔT 特征，捕捉不规则采样信息）
     - 缺失指示变量（Missing Indicator Matrix）
  3. 序列标准化（z-score，使用训练集统计量）
  4. 数据集划分（时间顺序，不随机打乱）
- **验收标准**:
  - 产出时序数据集 `sequential_dataset_train/val/test.pt`
  - 序列长度分布统计图
  - 不同预测窗口下的 APS 阳性率汇总

### T6.2 Bi-LSTM 模型优化与训练
- **依赖**: T6.1
- **复杂度**: ⭐⭐⭐⭐⭐
- **描述**:
  1. 网络架构改进：
     - 2层 Bi-LSTM（hidden_size=128）
     - 多头注意力机制（Multi-head Attention，heads=4）
     - Dropout（p=0.3）+ Layer Normalization
     - 全连接分类头（sigmoid输出）
  2. 训练策略：
     - 优化器：AdamW（weight_decay=1e-4）
     - 学习率调度：Cosine Annealing with Warm Restart
     - 早停（patience=15，监控验证集AUC）
     - 类别不平衡：Focal Loss（γ=2）
  3. 超参数搜索：Optuna 贝叶斯优化（50次trial）
  4. 5折时间感知交叉验证
- **验收标准**:
  - 验证集 AUC > 0.81
  - 训练曲线（loss/AUC vs epoch）图
  - 保存最优模型权重 `bilstm_best.pt`

### T6.3 Transformer 模型优化与训练
- **依赖**: T6.1
- **复杂度**: ⭐⭐⭐⭐⭐
- **描述**:
  1. 架构设计：
     - Positional Encoding（考虑不规则时间间隔，使用时间嵌入替代标准PE）
     - 3层 Transformer Encoder（d_model=128, nhead=8, dim_feedforward=256）
     - [CLS] Token 汇聚序列表示
     - Dropout（p=0.2）+ Pre-Layer Normalization
  2. 时间感知改进：
     - 将就诊间隔 ΔT 编码为连续嵌入，与特征拼接
     - Time2Vec 时间编码（sin+cos混合）
  3. 训练策略同 T6.2（Focal Loss + AdamW + Cosine LR）
  4. 与 Bi-LSTM 在相同测试集上公平对比
- **验收标准**:
  - 测试集 AUC > 0.81
  - 参数量与计算复杂度对比报告
  - 保存最优模型权重 `transformer_best.pt`

### T6.4 深度学习模型可解释性分析
- **依赖**: T6.2, T6.3
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. Transformer 注意力权重可视化：
     - 注意力热图（时间步 × 特征维度）
     - 多头注意力聚合（Head-averaged Attention）
     - 高风险患者 vs 低风险患者注意力模式对比
  2. Bi-LSTM 可解释性：
     - 梯度×输入（Gradient×Input）特征归因
     - 使用 Captum 库计算积分梯度（Integrated Gradients）
  3. 构建 SHAP DeepExplainer 对深度学习模型近似解释
  4. 与 Phase 3 SHAP 结果的特征重要性一致性对比
  5. 选取典型病例（真阳性/假阴性各2例）进行个体化时序归因可视化
- **验收标准**:
  - 产出 `Figure_Attention_Heatmap.pdf`
  - 产出 `Figure_DL_FeatureAttribution.pdf`
  - 深度学习特征归因排名表（与传统ML SHAP结果对比）

---

## Phase 7: 临床评分卡与综合报告

### T7.1 Nomogram 构建
- **依赖**: T3.4, T4.1
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 基于最优多因素逻辑回归模型构建个体化风险 Nomogram
  2. 纳入特征：≤8个（临床可操作性原则）
  3. 使用 `rms` 包（R/rpy2）或 Python `matplotlib` 手动绘制
  4. Nomogram 展示内容：
     - 每个预测因子的分值轴
     - 总分轴
     - APS预测概率轴（0~100%）
  5. 内部验证：Bootstrap C-index（B=1000）
  6. 校准曲线展示（apparent vs bias-corrected）
- **验收标准**:
  - 产出发表级 Nomogram `Figure_Nomogram.pdf`
  - Bootstrap C-index（期望值 > 0.80）
  - 校准曲线图 `Figure_Nomogram_Calibration.pdf`

### T7.2 床旁临床评分卡优化（EWS v2.0）
- **依赖**: T3.4, T4.2
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  1. 基于现有 EWS 评分卡（AUC=0.7118）进行以下优化：
     - 重新校准各指标分段截断值（使用 Youden index 最优截断）
     - 增加 LAC 阳性（+3分）和 B2GP1-IgG>40（+2分）条目
     - 调整权重：基于多因素 β 系数比例赋分（参照 SOFA/qSOFA范式）
  2. 简化版（无APL指标，5变量）vs 完整版（含APL，9变量）对比
  3. 临床实用性评估：
     - 检验者间一致性（Cohen's Kappa，模拟5名临床医生独立评分）
     - 评分时间（目标 <2分钟/患者）
  4. 网页计算器原型（HTML+JavaScript，离线可用）
- **验收标准**:
  - EWS v2.0 AUC > 0.74
  - 完整评分卡文档 `EWS_v2_scorecard.pdf`
  - 网页计算器 `ews_calculator.html`（本地可运行）

### T7.3 论文主图全套制作
- **依赖**: T2.3, T3.3, T4.1, T4.2, T5.1, T6.4, T7.1
- **复杂度**: ⭐⭐⭐⭐
- **描述**:
  按照目标期刊要求（ARD/A&R 图片规范），制作以下主图：

  **Paper 1 主图（6张）：**
  - Figure 1: 队列筛选流程图（CONSORT-style）
  - Figure 2: APL抗体谱系分析图（多阈值阳性率 + 抗体组合桑基图）
  - Figure 3: 单因素分析 Forest Plot（按临床维度分组）
  - Figure 4: SHAP Summary Plot（Top 20特征，beeswarm + bar双图）
  - Figure 5: 多模型 ROC 曲线对比 + 校准曲线（2-panel）
  - Figure 6: DCA 决策曲线 + Nomogram（2-panel）

  **Paper 2 主图（6张）：**
  - Figure 1: 纵向轨迹图 Panel（C3/C4/APTT/PLT × APS vs 对照，4-panel）
  - Figure 2: LCMM 潜在轨迹类别图
  - Figure 3: KM 生存曲线（APL四分层）+ Fine-Gray 竞争风险
  - Figure 4: 深度学习模型架构示意图（Transformer）
  - Figure 5: 注意力热图（高风险典型病例）
  - Figure 6: EWS v2.0 评分卡 + 风险分层流程图

  **通用要求：**
  - 分辨率 ≥ 300 DPI，TIFF/PDF 双格式
  - 字体：Arial 8pt（图内标注），10pt（标题）
  - 配色：色盲友好配色方案（viridis/ColorBrewer）
- **验收标准**:
  - 12张主图全部产出（PDF + TIFF）
  - 每张图附图例说明草稿
  - 通过色盲模拟检验（Coblis 或 `colorblind` 包验证）

### T7.4 补充材料制作
- **依赖**: T7.3
- **复杂度**: ⭐⭐⭐
- **描述**:
  **Paper 1 补充材料：**
  - Supplementary Table S1: 缺失数据模式分析
  - Supplementary Table S2: 多重插补 vs CCA 敏感性分析
  - Supplementary Table S3: 各模型超参数设置
  - Supplementary Table S4: 亚组分析 AUC 汇总
  - Supplementary Figure S1: 缺失数据热图
  - Supplementary Figure S2: 特征选择 Venn 图
  - Supplementary Figure S3: 嵌套CV各折AUC分布箱线图
  - Supplementary Figure S4: SHAP Dependence Plot（Top 6）

  **Paper 2 补充材料：**
  - Supplementary Table S1: 混合效应模型完整结果
  - Supplementary Table S2: LCMM 模型选择指标（BIC/AIC vs K）
  - Supplementary Table S3: Cox 单因素分析完整结果
  - Supplementary Figure S1: 时间依赖性 ROC 曲线
  - Supplementary Figure S2: Bi-LSTM vs Transformer 训练曲线
  - Supplementary Figure S3: 个体化时序归因图（典型病例）
- **验收标准**:
  - 所有补充材料产出（Word格式 + PDF）
  - 补充材料与正文交叉引用检查通过

---

## Phase 8: 论文撰写与投稿准备

### T8.1 Paper 1 草稿撰写
- **依赖**: T7.3, T7.4
- **复杂度**: ⭐⭐⭐⭐⭐
- **描述**:
  按 TRIPOD+AI 报告规范撰写（共约 3,500 词正文）：

  **论文结构：**
  1. **Title**: 简洁、包含研究设计类型和目标人群
     - 参考格式：*"A Machine Learning-based Risk Prediction Model for Antiphospholipid Syndrome in Systemic Lupus Erythematosus: Development and Internal Validation in a Longitudinal Cohort of 9,420 Patients"*
  2. **Abstract**（结构化，250词）: Background / Methods / Results / Conclusions
  3. **Introduction**（400词）: SLE-APS流行病学背景 → 现有预测工具的不足 → 本研究创新点
  4. **Methods**（1,000词）:
     - Study design & data source（引用TRIPOD条目1-3）
     - APS定义（引用2023 ACR/EULAR标准）
     - Statistical analysis（特征选择 → 模型训练 → 验证策略）
     - Missing data handling（引用多重插补细节）
  5. **Results**（1,200词）:
     - 队列特征（Table 1）
     - 单因素分析（Figure 3）
     - 模型性能（Figure 5）
     - 临床效用（Figure 6 DCA）
  6. **Discussion**（800词）:
     - 主要发现的临床意义
     - 与现有文献比较（引用 ≥ 5篇同类研究）
     - 局限性（至少4点，诚实表述）
  7. **Conclusion**（100词）
  8. **References**（Vancouver格式，40-60条）
- **验收标准**:
  - 完整草稿 `paper1_draft_v1.docx`（包含所有图表引用）
  - TRIPOD+AI checklist 完成度 ≥ 90%
  - 字数符合目标期刊要求（A&R: ≤4,000词）

### T8.2 Paper 2 草稿撰写
- **依赖**: T7.3, T7.4
- **复杂度**: ⭐⭐⭐⭐⭐
- **描述**:
  按 STROBE + TRIPOD+AI 报告规范撰写（共约 4,000 词正文）：

  **论文结构：**
  1. **Title**: 强调纵向轨迹和深度学习
     - 参考格式：*"Longitudinal Biomarker Trajectories and Deep Learning-based Early Warning System for Antiphospholipid Syndrome in SLE: A Multi-dimensional Time-series Analysis"*
  2. **Abstract**（结构化，300词）
  3. **Introduction**（500词）: 纵向预测的优势 → 时序模型在风湿病中的应用 → EWS临床需求
  4. **Methods**（1,200词）:
     - 时序数据构建
     - 混合效应模型方法
     - LCMM 方法描述
     - 深度学习架构（Bi-LSTM / Transformer）
     - EWS评分卡开发与验证
  5. **Results**（1,500词）:
     - 纵向轨迹差异（Figure 1 + Table）
     - LCMM轨迹分类（Figure 2）
     - 生存分析（Figure 3）
     - 深度学习性能（含注意力可视化，Figure 4-5）
     - EWS评分系统（Figure 6）
  6. **Discussion**（800词）
  7. **References**（50-70条）
- **验收标准**:
  - 完整草稿 `paper2_draft_v1.docx`
  - STROBE checklist 完成度 ≥ 85%
  - 字数符合目标期刊要求（ARD: ≤4,500词）

### T8.3 统计报告与可重复性包
- **依赖**: T8.1, T8.2
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 撰写完整统计分析计划（SAP）文档
  2. 整理可重复分析代码：
     - 所有脚本添加注释（中英双语）
     - 统一使用随机种子（seed=42）
     - 生成 `requirements.txt`（固定包版本）
     - 创建 `run_all.sh` 一键重现所有结果
  3. 数据可用性声明（去标识化数据共享声明）
  4. 开放科学实践：
     - 代码上传 GitHub（MIT License）
     - EWS 计算器部署（如适用）
- **验收标准**:
  - `requirements.txt` 和 `README.md` 完整
  - `run_all.sh` 端到端运行无报错
  - SAP 文档 `statistical_analysis_plan.pdf`

### T8.4 投稿准备
- **依赖**: T8.1, T8.2, T8.3
- **复杂度**: ⭐⭐⭐
- **描述**:
  1. 按目标期刊格式要求进行格式化：
     - **Paper 1 投稿顺序**：Arthritis & Rheumatology → Lupus Science & Medicine → Rheumatology
     - **Paper 2 投稿顺序**：Annals of the Rheumatic Diseases → RMD Open → Seminars in Arthritis and Rheumatism
  2. 撰写 Cover Letter（强调新颖性、临床价值、数据规模）
  3. 准备 Reviewer Response Template（为同行评审意见预留框架）
  4. 伦理声明（IRB批号）、利益冲突声明、作者贡献（CRediT格式）
  5. 图片格式最终检查（分辨率、颜色模式CMYK/RGB确认）
  6. 提交前清单（Submission Checklist）逐项确认
- **验收标准**:
  - 两篇论文均完成格式化，符合各目标期刊投稿指南
  - Cover Letter 草稿（各1封）
  - Submission Checklist 全部打钩
  - 电子投稿系统提交确认邮件（最终目标）

---

## 附录：任务依赖关系总图

T0.1 → T0.2 → T0.3
T0.1 → T0.4
↓
T0.4 → T1.1 → T1.2 → T2.1 → T2.2
↓
T2.3 → T3.1 → T3.2 → T3.3
→ T3.4 → T4.1 → T4.2 → T4.3 → T4.4
T1.3 → T5.1 → T5.2 ↓
→ T5.3 → T5.4 T7.1 → T7.2
T6.1 → T6.2 → T6.4 T7.3 → T7.4
→ T6.3 → T6.4 ↓
T8.1 → T8.3 → T8.4
T8.2 → T8.3 → T8.4