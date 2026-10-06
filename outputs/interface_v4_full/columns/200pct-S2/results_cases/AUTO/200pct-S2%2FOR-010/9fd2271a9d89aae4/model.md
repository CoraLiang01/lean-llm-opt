#### Abstract Mathematical Model

**Sets:**
- $S$: Set of store sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: Set of products, indexed by $p$ (ProductName from file_1_view_0)

**Parameters:**
- $C_s$: Display space capacity of section $s$ (Capacity from file_0_view_0)
- $v_p$: Price (revenue per unit) of product $p$ (Value from file_1_view_0)
- $w_p$: Shelf space requirement per unit of product $p$ (Weight from file_1_view_0)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Section capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]
- Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$ (Sections): SectionID from file_0_view_0 (capacity.csv)
- $C_s$: Capacity from file_0_view_0 (capacity.csv), mapped by SectionID
- $P$ (Products): ProductName from file_1_view_0 (products.csv)
- $v_p$: Value from file_1_view_0 (products.csv), mapped by ProductName
- $w_p$: Weight from file_1_view_0 (products.csv), mapped by ProductName

All parameters and indices are to be used exactly as they appear in the source files and columns.