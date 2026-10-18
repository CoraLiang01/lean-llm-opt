#### Abstract Mathematical Model

**Index Sets:**
- $\mathcal{I}$: Set of options (from OptionCharacteristics.csv), $|\mathcal{I}| = 120$
- $\mathcal{J}$: Set of assets (from Option_AssetReferenceMatrix.csv), $|\mathcal{J}| = 6$
- $\mathcal{G} = \{\Delta, \Gamma, \text{Vega}\}$: Set of Greeks

**Parameters:**
- $\text{Cost}_i$: Per-contract cost of option $i$, from OptionCharacteristics.csv, column "Cost", $i \in \mathcal{I}$
- $\text{Delta}_i$: Per-contract delta of option $i$, from OptionCharacteristics.csv, column "Delta", $i \in \mathcal{I}$
- $\text{Gamma}_i$: Per-contract gamma of option $i$, from OptionCharacteristics.csv, column "Gamma", $i \in \mathcal{I}$
- $\text{Vega}_i$: Per-contract vega of option $i$, from OptionCharacteristics.csv, column "Vega", $i \in \mathcal{I}$
- $\text{MaxLong}_i$: Maximum long position for option $i$, from OptionCharacteristics.csv, column "MaxLong", $i \in \mathcal{I}$
- $\text{MaxShort}_i$: Maximum short position for option $i$, from OptionCharacteristics.csv, column "MaxShort", $i \in \mathcal{I}$
- $A_{i,j}$: Binary indicator, 1 if option $i$ references asset $j$, 0 otherwise, from Option_AssetReferenceMatrix.csv, $i \in \mathcal{I},\ j \in \mathcal{J}$
- $G_{\text{initial}}^{(g)}$: Initial net exposure for Greek $g \in \mathcal{G}$ (given: $G_{\text{initial}}^{(\Delta)} = 0.25$, $G_{\text{initial}}^{(\Gamma)} = 0.08$, $G_{\text{initial}}^{(\text{Vega})} = 0.17$)
- $\text{Tolerance}_g$: Risk band tolerance for Greek $g \in \mathcal{G}$ (given: $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$)

**Decision Variables:**
- $x_i \in \mathbb{Z}$: Integer number of contracts for option $i$ (positive for long, negative for short), $\forall i \in \mathcal{I}$
- $y_i \geq 0$: Auxiliary variable for $|x_i|$, $\forall i \in \mathcal{I}$

**Objective:**
\[
\min \sum_{i \in \mathcal{I}} \text{Cost}_i \cdot y_i
\]

**Constraints:**

1. **Absolute Value Linking:**
   \[
   y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0,\quad \forall i \in \mathcal{I}
   \]

2. **Trading Limits:**
   \[
   \text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i,\quad \forall i \in \mathcal{I}
   \]

3. **Greek Exposure Constraints (for each $g \in \mathcal{G}$):**
   - Let $G_i^{(g)}$ denote the per-contract Greek $g$ for option $i$:
     - $G_i^{(\Delta)} = \text{Delta}_i$
     - $G_i^{(\Gamma)} = \text{Gamma}_i$
     - $G_i^{(\text{Vega})} = \text{Vega}_i$
   - For each $g \in \mathcal{G}$:
     \[
     \left| G_{\text{initial}}^{(g)} + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} G_i^{(g)} \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_g
     \]

**Data Mapping:**

- OptionCharacteristics.csv (table_id: file_0_view_0): columns "Option", "Cost", "Delta", "Gamma", "Vega", "MaxLong", "MaxShort"; all 120 rows used, no filtering.
- Option_AssetReferenceMatrix.csv (table_id: file_1_view_0): columns "Unnamed: 0" (option ID), "Asset_1", ..., "Asset_6"; all 120 rows used, no filtering.
- Initial net Greeks and risk tolerances: provided in user query, not from file.

**Notes:**
- All index sets, parameters, and constraints are defined symbolically and reference the exact columns and tables as returned by CSVQA.
- The model uses auxiliary variables $y_i$ to represent $|x_i|$ for the objective.
- The asset-reference matrix $A_{i,j}$ is used to aggregate each Greek exposure across all referenced assets for each option.
- All trading limits and per-contract Greeks are as given in the data, with no additional filtering or modification.