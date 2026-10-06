#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of sections (indexed by $s$), from `file_0_view_0.SectionID`
- $P$: Set of products (indexed by $p$), from `file_1_view_0.ProductName$

**Parameters:**
- $C_s$: Capacity of section $s$, from `file_0_view_0.Capacity`
- $v_p$: Value (price) of product $p$, from `file_1_view_0.Value`
- $w_p$: Space requirement (shelf space) of product $p$, from `file_1_view_0.Weight$

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ to stock in section $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Section Capacity Constraints:**  
   For each section $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

#### Data Mapping

- $S$ (sections): All `SectionID` values from `file_0_view_0` (capacity.csv)
- $P$ (products): All `ProductName` values from `file_1_view_0` (products.csv)
- $C_s$: `Capacity` from `file_0_view_0`, keyed by `SectionID`
- $v_p$: `Value` from `file_1_view_0`, keyed by `ProductName`
- $w_p$: `Weight` from `file_1_view_0`, keyed by `ProductName`

---

**All parameters, index sets, and constraints are mapped directly from the retrieved data, preserving original file and column names.**