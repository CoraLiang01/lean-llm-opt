## Abstract Mathematical Model

Let:
- $I$ = set of options, indexed by $i$ (from OptionCharacteristics.csv, column Option)
- $J$ = set of assets, indexed by $j$ (from Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6)
- $G$ = set of Greeks, $G = \{\Delta, \Gamma, \text{Vega}\}$

### Parameters

- $\text{Cost}_i$ = per-contract cost of option $i$ (OptionCharacteristics.csv, column Cost)
- $\text{Delta}_i$ = per-contract delta of option $i$ (OptionCharacteristics.csv, column Delta)
- $\text{Gamma}_i$ = per-contract gamma of option $i$ (OptionCharacteristics.csv, column Gamma)
- $\text{Vega}_i$ = per-contract vega of option $i$ (OptionCharacteristics.csv, column Vega)
- $\text{MaxLong}_i$ = maximum allowed long position for option $i$ (OptionCharacteristics.csv, column MaxLong)
- $\text{MaxShort}_i$ = maximum allowed short position for option $i$ (OptionCharacteristics.csv, column MaxShort)
- $A_{ij}$ = 1 if option $i$ references asset $j$, 0 otherwise (Option_AssetReferenceMatrix.csv, columns Asset_1,...,Asset_6)
- $G_{\text{initial}}$ = initial net exposure for Greek $G$ (given: $\Delta=0.25$, $\Gamma=0.08$, $\text{Vega}=0.17$)
- $\text{Tolerance}_G$ = risk band for Greek $G$ (given: $0.06$ for $\Delta$, $0.05$ for $\Gamma$, $0.07$ for Vega)

### Decision Variables

- $x_i \in \mathbb{Z}$: integer number of contracts for option $i$ (positive for long, negative for short)

- $y_i \geq 0$: auxiliary variable for $|x_i|$ (for absolute value in objective)

### Objective

Minimize total hedging cost:
$$
\min \sum_{i \in I} \text{Cost}_i \cdot y_i
$$

### Constraints

#### 1. Absolute Value Linking

For all $i \in I$:
$$
y_i \geq x_i
$$
$$
y_i \geq -x_i
$$

#### 2. Trading Limits

For all $i \in I$:
$$
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i
$$

#### 3. Risk Exposure Constraints

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
$$
\left| G_{\text{initial}} + \sum_{i \in I} \sum_{j \in J} G_i \cdot A_{ij} \cdot x_i \right| \leq \text{Tolerance}_G
$$
where $G_i$ is the per-contract value for Greek $G$ for option $i$ (i.e., $\text{Delta}_i$, $\text{Gamma}_i$, or $\text{Vega}_i$).

#### 4. Variable Domains

For all $i \in I$:
$$
x_i \in \mathbb{Z}
$$
$$
y_i \geq 0
$$

---

## Data Mapping

- $I$ (options): OptionCharacteristics.csv, column Option, table_id: file_0_view_0
- $J$ (assets): Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6, table_id: file_1_view_0
- $\text{Cost}_i$: OptionCharacteristics.csv, column Cost, table_id: file_0_view_0
- $\text{Delta}_i$: OptionCharacteristics.csv, column Delta, table_id: file_0_view_0
- $\text{Gamma}_i$: OptionCharacteristics.csv, column Gamma, table_id: file_0_view_0
- $\text{Vega}_i$: OptionCharacteristics.csv, column Vega, table_id: file_0_view_0
- $\text{MaxLong}_i$: OptionCharacteristics.csv, column MaxLong, table_id: file_0_view_0
- $\text{MaxShort}_i$: OptionCharacteristics.csv, column MaxShort, table_id: file_0_view_0
- $A_{ij}$: Option_AssetReferenceMatrix.csv, columns Asset_1,...,Asset_6, table_id: file_1_view_0, with row Option = Unnamed: 0
- $G_{\text{initial}}$: given in user description
- $\text{Tolerance}_G$: given in user description

---

## Index Alignment

- For each $i$, align OptionCharacteristics.csv Option (file_0_view_0, column Option) with Option_AssetReferenceMatrix.csv Unnamed: 0 (file_1_view_0, column Unnamed: 0).

---

## Summary

This model minimizes total hedging cost (sum of per-contract costs times absolute value of positions), subject to:
- Integer trading limits for each option,
- Risk exposures for each Greek (Delta, Gamma, Vega) after hedging, aggregated over all referenced assets, must be within the specified tolerance bands,
- All data and mappings are explicitly referenced by table and column as required.