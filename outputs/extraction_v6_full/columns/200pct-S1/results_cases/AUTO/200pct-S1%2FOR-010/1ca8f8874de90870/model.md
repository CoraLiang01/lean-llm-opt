#### Abstract Mathematical Model

**Sets:**
- $S$: set of sections (indexed by $s$), from file_0_view_0, column SectionID
- $P$: set of products (indexed by $p$), from file_1_view_0, column ProductName

**Parameters:**
- $c_s$: display space capacity of section $s$ (from file_0_view_0, Capacity)
- $v_p$: price of product $p$ (from file_1_view_0, Value)
- $w_p$: shelf space requirement of product $p$ (from file_1_view_0, Weight)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Section capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s
\]
- Integrality and nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- $S$ (sections): file_0_view_0, column SectionID
- $P$ (products): file_1_view_0, column ProductName
- $c_s$: file_0_view_0, Capacity, indexed by SectionID
- $v_p$: file_1_view_0, Value, indexed by ProductName
- $w_p$: file_1_view_0, Weight, indexed by ProductName

All indices, parameters, and constraints are mapped directly to the original file columns and business identifiers as returned by CSVQA.