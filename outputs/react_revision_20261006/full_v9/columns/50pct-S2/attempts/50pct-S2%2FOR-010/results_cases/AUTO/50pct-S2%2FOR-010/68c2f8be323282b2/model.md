Mathematical Optimization Model

Index Sets:
- $S$: Set of sections, indexed by $s$ (from file_0_view_0, column SectionID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $c_s$: Capacity (display space limit) of section $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value (price) of product $p$ (from file_1_view_0, column Value)
- $w_p$: Space requirement (shelf space) of product $p$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:

Section Capacity Constraints:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
\]

Integrality and Nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping:

- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName
- $c_s$: file_0_view_0, column Capacity, keyed by SectionID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{sp}$: Decision variable for units of product $p$ in section $s$ (indexed by SectionID and ProductName)