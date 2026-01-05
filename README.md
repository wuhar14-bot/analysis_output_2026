# OLIF Citation Analysis (2008-2026)

OLIF (Oblique Lumbar Interbody Fusion) 文献计量分析项目 - 书目耦合与研究流派识别

## 项目概述

本项目使用 **书目耦合 (Bibliographic Coupling)** 和 **PCA主成分分析** 方法，从Scopus数据库中识别OLIF研究领域的主要研究流派。

### 数据来源
- **数据库**: Scopus (HKU Libraries)
- **时间范围**: 2008-2026
- **检索式**:
  ```
  TITLE-ABS-KEY("oblique lumbar interbody fusion" OR "OLIF" OR "oblique lateral interbody fusion")
  AND PUBYEAR > 2007 AND PUBYEAR < 2027
  ```
- **文献数量**: 958篇 (筛选后750篇英文Article/Review)

---

## 分析流程

### Step 1: 数据加载与筛选
```
原始数据 (958篇)
    ↓ 筛选英文文献
    ↓ 筛选 Article/Review
    ↓ 筛选有参考文献的文献
过滤后 (750篇)
```

### Step 2: 引用匹配 (倒排索引加速)

**核心优化**: 使用倒排词索引实现O(1)查找，替代O(n²)暴力匹配

```python
# 构建倒排索引
title_words = {}
for title in titles:
    words = [w for w in re.split(r'\W+', title) if len(w) > 3][:5]
    for word in words:
        title_words[word].add(title)

# 匹配时O(1)查找
for word in reference_words:
    if word in title_words:
        candidates.update(title_words[word])
```

**效果**: 匹配时间从数分钟降至 ~0.8秒

### Step 3: 邻接矩阵构建

构建 **文献-引用** 二元矩阵 A:
- `A[i,j] = 1` 表示文献i引用了文献j
- 矩阵大小: 641 × 641

### Step 4: 书目耦合计算

**书目耦合 (Bibliographic Coupling)**:
```
BC = A × Aᵀ
```
- `BC[i,j]` = 文献i和文献j共同引用的参考文献数量
- 共同引用越多，研究主题越相似

### Step 5: 余弦相似度标准化

```python
sim = cosine_similarity(BC)
```
- 将耦合强度标准化到 [0, 1] 区间
- 消除文献引用数量差异的影响

### Step 6: PCA主成分分析

使用 **Kaiser准则** (特征值 > 1) 确定因子数:
```python
n_factors = np.count_nonzero(eigenvalues > 1)  # 结果: 15
```

**因子分析**:
```python
fa = FactorAnalyzer(n_factors=15, rotation='promax')
fa.fit(similarity_matrix)
```

### Step 7: 研究流派提取

以载荷阈值 > 0.3 分配文献到各流派:
```python
for factor in loadings.columns:
    papers = loadings[loadings[factor] > 0.3]
    papers.to_csv(f'{factor}.csv')
```

---

## 分析结果

### 数据统计

| 指标 | 数值 |
|------|------|
| Scopus检索结果 | 958篇 |
| 英文Article/Review | 750篇 |
| 有效引用关系对 | 2,113对 |
| 核心引用文献 | 572篇 |
| 研究因子数 | 15个 |
| 方差解释率 | 99.1% |

### 研究流派识别

| 流派 | 文献数 | 占比 | 研究主题 |
|------|--------|------|----------|
| **F1** | 255篇 | 44.6% | OLIF核心技术与临床应用 |
| **F2** | 111篇 | 19.4% | 疗效评估与安全性研究 |
| F3 | 43篇 | 7.5% | 特定适应症/并发症 |
| F4 | 18篇 | 3.1% | 间接减压与椎管狭窄 |
| F5 | 27篇 | 4.7% | 内固定策略与技术优化 |
| F6 | 15篇 | 2.6% | 影像学评估与特殊适应症 |
| F7 | 11篇 | 1.9% | 生物力学与L5-S1问题 |
| F8 | 7篇 | 1.2% | 罕见并发症与解剖安全 |
| F9 | 24篇 | 4.2% | 术式比较与远期疗效 |
| F10 | 28篇 | 4.9% | 系统综述与Meta分析 |
| F11 | 10篇 | 1.7% | 邻近节段退变与融合率研究 |
| F12 | 6篇 | 1.0% | 导航辅助与微创技术 |
| F13 | 28篇 | 4.9% | 内固定优化与骨质疏松管理 |
| F14 | 23篇 | 4.0% | 单体位手术与特殊适应症 |
| F15 | 24篇 | 4.2% | L5-S1入路挑战与骨质量评估 |

---

## 文件结构

```
analysis_output_2026/
├── README.md                 # 本文档
├── analysis_overview.html    # HTML可视化报告
├── scree_plot.png            # 碎石图
├── factor_loadings.csv       # PCA载荷矩阵 (572×15)
├── factor_overlap.csv        # 因子间重叠矩阵
├── F1.csv ~ F15.csv          # 各研究流派文献列表
│
├── data/
│   └── scopus_2008_2026.csv  # Scopus原始数据
│
└── scripts/
    ├── olif_fast_analysis.py  # 主分析脚本 (优化版)
    └── olif_analysis_2026.py  # 备用脚本
```

---

## 依赖环境

```bash
pip install numpy pandas matplotlib scikit-learn factor-analyzer
```

## 运行方法

```bash
cd E:\claude-code\Zhao\analysis_output_2026\scripts
python olif_fast_analysis.py
```

---

## 与旧版本对比 (2012-2022 vs 2008-2026)

| 指标 | 旧版 (2012-2022) | 新版 (2008-2026) | 变化 |
|------|------------------|------------------|------|
| 文献数量 | 345篇 | 750篇 | +117% |
| 引用对 | 9,283对 | 2,113对 | -77%* |
| 分析文献 | 333篇 | 572篇 | +72% |
| 研究因子 | 14个 | 15个 | +1 |

*引用对减少是因为新版使用更严格的标题匹配算法，提高精确度

---

## 更新日志

### 2026-01-04
- 更新数据至2008-2026年范围
- 优化引用匹配算法 (倒排索引, O(1)查找)
- 使用PCA替代因子分析 (避免奇异矩阵问题)
- 生成中文HTML分析报告
- 识别并标注15个研究流派的具体主题

---

## 作者

Hao Wu
Department of Orthopaedics & Traumatology
The University of Hong Kong
