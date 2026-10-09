ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

Parameters:
- $C_s$: capacity of section $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: value (revenue) per unit of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: space requirement per unit of product $p \in P$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName
- $C_s$: file_0_view_0, columns SectionID, Capacity
- $v_p$: file_1_view_0, columns ProductName, Value
- $w_p$: file_1_view_0, columns ProductName, Weight
- $x_{sp}$: for all $s \in S$, $p \in P$ (decision variables as described above)