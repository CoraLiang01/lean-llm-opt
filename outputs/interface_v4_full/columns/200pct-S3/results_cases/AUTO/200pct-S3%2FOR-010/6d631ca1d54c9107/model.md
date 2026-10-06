## Abstract Mathematical Model

**Index Sets:**
- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

**Parameters:**
- $c_s$: capacity of section $s$ (from file_0_view_0, column Capacity, key SectionID)
- $v_p$: value (price) of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: shelf space requirement of product $p$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Section Capacity Constraints:**  
   For each section $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

## Data Mapping

- $S$ (sections): file_0_view_0, column SectionID
- $c_s$: file_0_view_0, column Capacity, key SectionID
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

All data is used as returned, preserving file and row order. No columns or records are omitted.