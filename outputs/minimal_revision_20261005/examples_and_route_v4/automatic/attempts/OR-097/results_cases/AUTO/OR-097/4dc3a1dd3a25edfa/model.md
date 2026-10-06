**Abstract Mathematical Model**

**Index Sets**
- $\mathcal{I}$: Set of options (Option), from OptionCharacteristics.csv.
- $\mathcal{J}$: Set of assets (Asset_1, ..., Asset_6), from Option_AssetReferenceMatrix.csv.
- $\mathcal{G}$: Set of Greeks, $\{\Delta, \Gamma, \text{Vega}\}$.

**Parameters**
- $\text{Cost}_i$: Per-contract cost of option $i$, from OptionCharacteristics.csv.
- $\text{Delta}_i$: Per-contract delta of option $i$, from OptionCharacteristics.csv.
- $\text{Gamma}_i$: Per-contract gamma of option $i$, from OptionCharacteristics.csv.
- $\text{Vega}_i$: Per-contract vega of option $i$, from OptionCharacteristics.csv.
- $\text{MaxLong}_i$: Maximum long position for option $i$, from OptionCharacteristics.csv.
- $\text{MaxShort}_i$: Maximum short position for option $i$, from OptionCharacteristics.csv.
- $A_{ij}$: Binary, 1 if option $i$ references asset $j$, 0 otherwise, from Option_AssetReferenceMatrix.csv.
- $G_{\text{initial}}$: Initial net exposure for Greek $G$ (given: $\Delta=0.25$, $\Gamma=0.08$, $\text{Vega}=0.17$).
- $\text{Tolerance}_G$: Risk band for Greek $G$ (given: $|\Delta| \le 0.06$, $|\Gamma| \le 0.05$, $|\text{Vega}| \le 0.07$).

**Decision Variables**
- $x_i \in \mathbb{Z}$: Integer number of contracts for option $i$ (positive for long, negative for short), $\forall i \in \mathcal{I}$.
- $y_i \ge 0$: Auxiliary variable for $|x_i|$, $\forall i \in \mathcal{I}$.

**Objective**
\[
\min \sum_{i \in \mathcal{I}} \text{Cost}_i \cdot y_i
\]

**Constraints**

1. **Absolute Value Linearization**
   \[
   y_i \ge x_i,\quad y_i \ge -x_i,\quad \forall i \in \mathcal{I}
   \]

2. **Trading Limits**
   \[
   \text{MaxShort}_i \le x_i \le \text{MaxLong}_i,\quad \forall i \in \mathcal{I}
   \]

3. **Greek Risk Constraints** (for each $G \in \mathcal{G}$)
   \[
   \left| G_{\text{initial}} + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} G_i \cdot A_{ij} \cdot x_i \right| \le \text{Tolerance}_G
   \]
   where $G_i$ is $\text{Delta}_i$, $\text{Gamma}_i$, or $\text{Vega}_i$ for $G = \Delta, \Gamma, \text{Vega}$, respectively.

   Equivalently, for each $G \in \mathcal{G}$:
   \[
   -\text{Tolerance}_G \le G_{\text{initial}} + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} G_i \cdot A_{ij} \cdot x_i \le \text{Tolerance}_G
   \]

**Data Mapping**

- $\mathcal{I}$: All Option values from OptionCharacteristics.csv, column Option, table_id=file_0_view_0.
- $\mathcal{J}$: All asset columns Asset_1, ..., Asset_6 from Option_AssetReferenceMatrix.csv, table_id=file_1_view_0.
- $\text{Cost}_i$, $\text{Delta}_i$, $\text{Gamma}_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$: from OptionCharacteristics.csv, columns Cost, Delta, Gamma, Vega, MaxLong, MaxShort, table_id=file_0_view_0, keyed by Option.
- $A_{ij}$: from Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6, table_id=file_1_view_0, keyed by Option (OptionCharacteristics.csv:Option = Option_AssetReferenceMatrix.csv:Unnamed: 0).
- $G_{\text{initial}}$: Query-provided values: $\Delta=0.25$, $\Gamma=0.08$, $\text{Vega}=0.17$.
- $\text{Tolerance}_G$: Query-provided values: $|\Delta| \le 0.06$, $|\Gamma| \le 0.05$, $|\text{Vega}| \le 0.07$.

**Variable Domains**
- $x_i \in \mathbb{Z}$, $\forall i \in \mathcal{I}$
- $y_i \ge 0$, $\forall i \in \mathcal{I}$

---

**Summary:**  
Minimize total hedging cost over integer option positions, subject to per-option trading limits and portfolio Greek exposures (Delta, Gamma, Vega) after hedging being within specified bands, using the asset-reference matrix to aggregate exposures. All data and index sets are mapped directly from the supplied files and query.