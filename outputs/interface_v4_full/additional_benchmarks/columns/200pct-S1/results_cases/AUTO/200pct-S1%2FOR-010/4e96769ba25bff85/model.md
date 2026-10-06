#### Abstract Mathematical Model

**Sets:**
- $S$: Set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of section $s$ (from file_0_view_0, column Capacity, key SectionID)
- $v_p$: Value (price) of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: Shelf space requirement of product $p$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ to be placed in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Section capacity constraints:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
\]
- Integrality and nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- **SectionID** (file_0_view_0, column SectionID): $S$
- **Capacity** (file_0_view_0, column Capacity): $C_s$, keyed by SectionID
- **ProductName** (file_1_view_0, column ProductName): $P$
- **Value** (file_1_view_0, column Value): $v_p$, keyed by ProductName
- **Weight** (file_1_view_0, column Weight): $w_p$, keyed by ProductName

All other columns are ignored as unrelated to the optimization model.