#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of options (from OptionCharacteristics.csv), $|I|=120$
- $J$: set of assets (from Option_AssetReferenceMatrix.csv), $|J|=6$
- $G$: set of Greeks, $G = \{\Delta, \Gamma, \text{Vega}\}$

**Parameters:**
- $\text{Cost}_i$: per-contract cost of option $i$, from OptionCharacteristics.csv, $i \in I$
- $\text{Delta}_i$: per-contract delta of option $i$, from OptionCharacteristics.csv, $i \in I$
- $\text{Gamma}_i$: per-contract gamma of option $i$, from OptionCharacteristics.csv, $i \in I$
- $\text{Vega}_i$: per-contract vega of option $i$, from OptionCharacteristics.csv, $i \in I$
- $\text{MaxLong}_i$: maximum long position for option $i$, from OptionCharacteristics.csv, $i \in I$
- $\text{MaxShort}_i$: maximum short position for option $i$, from OptionCharacteristics.csv, $i \in I$
- $A_{i,j}$: binary, $=1$ if option $i$ references asset $j$, $0$ otherwise, from Option_AssetReferenceMatrix.csv, $i \in I, j \in J$
- $G_{\text{initial},j}$: initial net exposure for Greek $G$ on asset $j$, given for each $G \in \{\Delta, \Gamma, \text{Vega}\}$ and $j \in J$
    - For this instance: $G_{\text{initial},j}$ is the same for all $j$ (i.e., $\Delta_{\text{initial}}=0.25$, $\Gamma_{\text{initial}}=0.08$, $\text{Vega}_{\text{initial}}=0.17$)
- $\text{Tolerance}_G$: risk band for Greek $G$, given as $|\Delta| \leq 0.06$, $|\Gamma| \leq 0.05$, $|\text{Vega}| \leq 0.07$

**Decision Variables:**
- $x_i \in \mathbb{Z}$: number of contracts for option $i$ (positive for long, negative for short), $i \in I$
- $z_i \geq 0$: auxiliary variable for $|x_i|$, $i \in I$

**Objective:**
\[
\min \sum_{i \in I} \text{Cost}_i \cdot z_i
\]

**Constraints:**

1. **Absolute Value Linking:**
   \[
   z_i \geq x_i,\quad z_i \geq -x_i,\quad z_i \geq 0,\quad \forall i \in I
   \]

2. **Trading Limits:**
   \[
   \text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i,\quad \forall i \in I
   \]

3. **Greek Risk Band Constraints (for each Greek $G$ and each asset $j$):**
   \[
   \left| G_{\text{initial},j} + \sum_{i \in I} G_i \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_G,\quad \forall G \in \{\Delta, \Gamma, \text{Vega}\},\ \forall j \in J
   \]
   where $G_i$ is the per-contract value for Greek $G$ for option $i$ (i.e., $\text{Delta}_i$, $\text{Gamma}_i$, or $\text{Vega}_i$).

   Equivalently, for each $G$ and $j$:
   \[
   -\text{Tolerance}_G \leq G_{\text{initial},j} + \sum_{i \in I} G_i \cdot A_{i,j} \cdot x_i \leq \text{Tolerance}_G
   \]

4. **Variable Domains:**
   \[
   x_i \in \mathbb{Z},\quad \forall i \in I
   \]
   \[
   z_i \geq 0,\quad \forall i \in I
   \]

---

**Data Mapping:**

- OptionCharacteristics.csv: columns Option, Cost, Delta, Gamma, Vega, MaxLong, MaxShort; all rows; table_id: file_0_view_0
- Option_AssetReferenceMatrix.csv: columns Unnamed: 0 (option identifier), Asset_1, Asset_2, Asset_3, Asset_4, Asset_5, Asset_6; all rows; table_id: file_1_view_0
- Initial net Delta, Gamma, Vega and risk band tolerances: provided in user query

---

**Notes:**
- All data from both files is used without filtering.
- The model is fully integer, with absolute value handled via auxiliary variables $z_i$.
- Each Greek constraint is enforced for each asset, using the asset-reference matrix.
- The initial exposures and tolerances are applied identically for each asset as per the query.