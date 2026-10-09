Mathematical Model

Index Sets:
- $S$: set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: set of products, indexed by $p$ (ProductName from file_1_view_0)

Parameters:
- $c_s$: display space capacity of section $s$ (Capacity from file_0_view_0)
- $v_p$: value (price) of product $p$ (Value from file_1_view_0)
- $w_p$: shelf space requirement of product $p$ (Weight from file_1_view_0)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to stock in section $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Constraints:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $c_s$: Capacity from file_0_view_0, indexed by SectionID
- $v_p$: Value from file_1_view_0, indexed by ProductName
- $w_p$: Weight from file_1_view_0, indexed by ProductName
- $x_{sp}$: number of units of product $p$ to stock in section $s$ (decision variable, integer, nonnegative)