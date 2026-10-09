Mathematical Model

Index Sets:
- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

Parameters:
- $c_s$: capacity of section $s$ (from file_0_view_0, column Capacity, key SectionID)
- $v_p$: value (revenue) per unit of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: space requirement per unit of product $p$ (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName
- $c_s$: file_0_view_0, column Capacity, key SectionID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName