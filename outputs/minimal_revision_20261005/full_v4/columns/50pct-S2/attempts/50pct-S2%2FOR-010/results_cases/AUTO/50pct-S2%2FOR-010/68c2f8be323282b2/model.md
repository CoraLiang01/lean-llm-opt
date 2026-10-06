**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

**Parameters:**
- $c_s$: capacity of section $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: value (price) of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: shelf space requirement of product $p \in P$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to be placed in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

**Data Mapping**

- $S$: All SectionID values from file_0_view_0, column SectionID
- $P$: All ProductName values from file_1_view_0, column ProductName
- $c_s$: file_0_view_0, columns SectionID (key), Capacity (value)
- $v_p$: file_1_view_0, columns ProductName (key), Value (value)
- $w_p$: file_1_view_0, columns ProductName (key), Weight (value)