#### Abstract Mathematical Model

**Index Sets:**
- $\mathcal{I}$: Set of options (from OptionCharacteristics.csv, column Option)
- $\mathcal{J}$: Set of assets (from Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6)
- $\mathcal{G} = \{\Delta, \Gamma, \text{Vega}\}$: Set of Greeks

**Parameters:**
- $\text{Cost}_i$: Per-contract cost of option $i$ (OptionCharacteristics.csv, column Cost)
- $\text{Delta}_i$: Per-contract delta of option $i$ (OptionCharacteristics.csv, column Delta)
- $\text{Gamma}_i$: Per-contract gamma of option $i$ (OptionCharacteristics.csv, column Gamma)
- $\text{Vega}_i$: Per-contract vega of option $i$ (OptionCharacteristics.csv, column Vega)
- $\text{MaxLong}_i$: Maximum allowed long position for option $i$ (OptionCharacteristics.csv, column MaxLong)
- $\text{MaxShort}_i$: Maximum allowed short position for option $i$ (OptionCharacteristics.csv, column MaxShort)
- $A_{i,j}$: Binary indicator, 1 if option $i$ references asset $j$, 0 otherwise (Option_AssetReferenceMatrix.csv, columns Asset_1,...,Asset_6)
- $G_{\text{initial}}$: Initial net exposure for Greek $G$ (given: $\Delta$: 0.25, $\Gamma$: 0.08, Vega: 0.17)
- $\text{Tolerance}_G$: Risk band for Greek $G$ (given: $|\Delta| \leq 0.06$, $|\Gamma| \leq 0.05$, $|\text{Vega}| \leq 0.07$)

**Variables:**
- $x_i \in \mathbb{Z}$: Integer number of contracts for option $i$ (positive for long, negative for short), $\forall i \in \mathcal{I}$
- $z_i \geq 0$: Auxiliary variable for $|x_i|$, $\forall i \in \mathcal{I}$

**Objective:**
\[
\min \sum_{i \in \mathcal{I}} \text{Cost}_i \cdot z_i
\]

**Constraints:**

1. **Absolute Value Linking:**
   \[
   z_i \geq x_i,\quad z_i \geq -x_i,\quad \forall i \in \mathcal{I}
   \]

2. **Trading Limits:**
   \[
   \text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i,\quad \forall i \in \mathcal{I}
   \]

3. **Greek Risk Band Constraints:**  
   For each $G \in \mathcal{G}$,
   \[
   \left| G_{\text{initial}} + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} G_i \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_G
   \]
   where $G_i$ is $\text{Delta}_i$, $\text{Gamma}_i$, or $\text{Vega}_i$ for $G = \Delta, \Gamma, \text{Vega}$, respectively.

   Equivalently, for each $G \in \mathcal{G}$,
   \[
   -\text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} G_i \cdot A_{i,j} \cdot x_i \leq \text{Tolerance}_G
   \]

4. **Integrality:**
   \[
   x_i \in \mathbb{Z},\quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- $\mathcal{I}$: OptionCharacteristics.csv, column Option
- $\mathcal{J}$: Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6
- $\text{Cost}_i$: OptionCharacteristics.csv, column Cost
- $\text{Delta}_i$: OptionCharacteristics.csv, column Delta
- $\text{Gamma}_i$: OptionCharacteristics.csv, column Gamma
- $\text{Vega}_i$: OptionCharacteristics.csv, column Vega
- $\text{MaxLong}_i$: OptionCharacteristics.csv, column MaxLong
- $\text{MaxShort}_i$: OptionCharacteristics.csv, column MaxShort
- $A_{i,j}$: Option_AssetReferenceMatrix.csv, columns Asset_1, ..., Asset_6 (row Option $i$, column Asset $j$)
- $G_{\text{initial}}$: Provided in user query
- $\text{Tolerance}_G$: Provided in user query

---

**All sets, parameters, and variables are defined symbolically and mapped to their exact source columns.**