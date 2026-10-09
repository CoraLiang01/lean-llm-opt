Mathematical Optimization Model

Index Sets:
- $S$: Set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: Set of products, indexed by $p$ (ProductName from file_1_view_0)

Parameters:
- $c_s$: Capacity of section $s$ (Capacity from file_0_view_0, indexed by SectionID)
- $v_p$: Value (price) of product $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: Space requirement (Weight) of product $p$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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
- $S$: SectionID from file_0_view_0
- $P$: ProductName from file_1_view_0
- $c_s$: Capacity from file_0_view_0, indexed by SectionID
- $v_p$: Value from file_1_view_0, indexed by ProductName
- $w_p$: Weight from file_1_view_0, indexed by ProductName
- $x_{sp}$: Number of units of product $p$ to stock in section $s$ (decision variable, integer, nonnegative)