Mathematical Model

Sets:
- $S$: set of sections (indexed by $s$), from file_0_view_0[SectionID]
- $P$: set of products (indexed by $p$), from file_1_view_0[ProductName]

Parameters:
- $C_s$: capacity of section $s$, from file_0_view_0[Capacity]
- $v_p$: value (price) of product $p$, from file_1_view_0[Value]
- $w_p$: space requirement (weight) of product $p$, from file_1_view_0[Weight]

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

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

Data Mapping

- $S$: file_0_view_0[SectionID]
- $P$: file_1_view_0[ProductName]
- $C_s$: file_0_view_0[Capacity], keyed by SectionID $s$
- $v_p$: file_1_view_0[Value], keyed by ProductName $p$
- $w_p$: file_1_view_0[Weight], keyed by ProductName $p$
- $x_{sp}$: number of units of product $p$ in section $s$ (decision variable, indexed by $s \in S$, $p \in P$)