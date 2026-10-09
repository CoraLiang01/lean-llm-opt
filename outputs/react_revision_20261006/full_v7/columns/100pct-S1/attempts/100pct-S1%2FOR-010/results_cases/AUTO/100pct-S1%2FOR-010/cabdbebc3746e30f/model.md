#### Mathematical Model

Let:
- $S$ = set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$ = set of products, indexed by $p$ (ProductName from file_1_view_0)

Parameters:
- $C_s$ = capacity of section $s$ (Capacity from file_0_view_0)
- $v_p$ = value (revenue) per unit of product $p$ (Value from file_1_view_0)
- $w_p$ = space requirement per unit of product $p$ (Weight from file_1_view_0)

Decision variables:
- $x_{sp}$ = number of units of product $p$ to stock in section $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- $S$: All SectionID in file_0_view_0 (capacity.csv), column SectionID
- $P$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by SectionID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{sp}$: Decision variable for each $(s,p)$ pair

All indices, parameters, and mappings are defined exactly as in the returned data.