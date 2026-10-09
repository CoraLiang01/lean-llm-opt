ABSTRACT MATHEMATICAL MODEL

Sets:
- $S$: set of sections (indexed by $s$), with SectionID from file_0_view_0.
- $P$: set of products (indexed by $p$), with ProductName from file_1_view_0.

Parameters:
- $c_s$: capacity of section $s$ (Capacity column in file_0_view_0, indexed by SectionID).
- $v_p$: value (revenue) per unit of product $p$ (Value column in file_1_view_0, indexed by ProductName).
- $w_p$: space requirement per unit of product $p$ (Weight column in file_1_view_0, indexed by ProductName).

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$.

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping:
- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $c_s$: file_0_view_0, Capacity, indexed by SectionID
- $v_p$: file_1_view_0, Value, indexed by ProductName
- $w_p$: file_1_view_0, Weight, indexed by ProductName
- $x_{sp}$: number of units of product $p$ in section $s$ (decision variable, integer, nonnegative)