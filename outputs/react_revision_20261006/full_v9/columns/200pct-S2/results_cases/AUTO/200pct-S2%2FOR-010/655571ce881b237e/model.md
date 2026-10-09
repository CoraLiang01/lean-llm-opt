### Mathematical Model

**Index Sets:**
- $S$: set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $c_s$: capacity of section $s$ (from file_0_view_0, column Capacity)
- $v_p$: value (price) of product $p$ (from file_1_view_0, column Value)
- $w_p$: space requirement of product $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

### Data Mapping

- $S$: All SectionID in file_0_view_0 (capacity.csv), column SectionID
- $P$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_s$: file_0_view_0 (capacity.csv), column Capacity, keyed by SectionID
- $v_p$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_p$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $x_{sp}$: Decision variable for each $(s,p)$ pair

All parameters and index sets are defined by the current returned rows and columns as above.