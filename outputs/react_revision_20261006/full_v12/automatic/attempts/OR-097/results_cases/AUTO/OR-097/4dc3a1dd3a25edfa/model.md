## Symbolic Mathematical Model

**Sets:**
- $I = \{1, \dots, 120\}$: option indices (from OptionCharacteristics.csv, Option)
- $J = \{1, \dots, 6\}$: asset indices (from Option_AssetReferenceMatrix.csv, Asset_1 to Asset_6)

**Parameters (from data):**
- $\text{Cost}_i$: per-contract cost of option $i$ (file_0_view_0, Cost)
- $\Delta_i$: per-contract delta of option $i$ (file_0_view_0, Delta)
- $\Gamma_i$: per-contract gamma of option $i$ (file_0_view_0, Gamma)
- $\text{Vega}_i$: per-contract vega of option $i$ (file_0_view_0, Vega)
- $\text{MaxLong}_i$: max long position for option $i$ (file_0_view_0, MaxLong)
- $\text{MaxShort}_i$: max short position for option $i$ (file_0_view_0, MaxShort)
- $A_{ij}$: 1 if option $i$ references asset $j$, 0 otherwise (file_1_view_0, Asset_1 to Asset_6)
- $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$
- $\text{Tol}_\Delta = 0.06$, $\text{Tol}_\Gamma = 0.05$, $\text{Tol}_\text{Vega} = 0.07$

**Variables:**
- $x_i \in \mathbb{Z}$: integer number of contracts for option $i$ (positive for long, negative for short)
- $z_i \ge 0$: auxiliary variable for $|x_i|$ (for all $i \in I$)

---

**Objective:**
\[
\min \sum_{i \in I} \text{Cost}_i \cdot z_i
\]

**Subject to:**

**Absolute value linking:**
\[
z_i \ge x_i,\quad z_i \ge -x_i \qquad \forall i \in I
\]

**Trading limits:**
\[
\text{MaxShort}_i \le x_i \le \text{MaxLong}_i \qquad \forall i \in I
\]

**Risk constraints (for each Greek):**

- **Delta:**
\[
\left| \Delta_{\text{init}} + \sum_{i \in I} \sum_{j \in J} \Delta_i \cdot A_{ij} \cdot x_i \right| \le \text{Tol}_\Delta
\]

- **Gamma:**
\[
\left| \Gamma_{\text{init}} + \sum_{i \in I} \sum_{j \in J} \Gamma_i \cdot A_{ij} \cdot x_i \right| \le \text{Tol}_\Gamma
\]

- **Vega:**
\[
\left| \text{Vega}_{\text{init}} + \sum_{i \in I} \sum_{j \in J} \text{Vega}_i \cdot A_{ij} \cdot x_i \right| \le \text{Tol}_\text{Vega}
\]

**Variable domains:**
\[
x_i \in \mathbb{Z},\quad z_i \ge 0 \qquad \forall i \in I
\]

---

## Data Mapping

- $I$: file_0_view_0, Option (row order $i=1,\dots,120$)
- $J$: file_1_view_0, Asset_1 to Asset_6 (columns $j=1,\dots,6$)
- $\text{Cost}_i$, $\Delta_i$, $\Gamma_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$: file_0_view_0, columns as named, for each $i$
- $A_{ij}$: file_1_view_0, Option = file_0_view_0 Option, Asset_1 to Asset_6 as $j=1,\dots,6$
- Initial Greeks and tolerances: as given in the question

---

**All constraints, sets, and parameters are mapped directly from the provided data.**