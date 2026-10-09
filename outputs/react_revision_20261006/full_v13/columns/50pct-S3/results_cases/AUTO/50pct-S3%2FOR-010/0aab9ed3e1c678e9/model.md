Mathematical Model

Index Sets:
- $S$: set of sections (SectionID from file_0_view_0)
- $P$: set of products (ProductName from file_1_view_0)

Parameters:
- $c_s$: capacity of section $s$ (Capacity from file_0_view_0, indexed by SectionID)
- $v_p$: value (price) of product $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: space requirement of product $p$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

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

Data Mapping

- $S$: SectionID from file_0_view_0
- $P$: ProductName from file_1_view_0
- $c_s$: Capacity from file_0_view_0, indexed by SectionID
- $v_p$: Value from file_1_view_0, indexed by ProductName
- $w_p$: Weight from file_1_view_0, indexed by ProductName
- $x_{sp}$: number of units of product $p$ to stock in section $s$ (decision variable)