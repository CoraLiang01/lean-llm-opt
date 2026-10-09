ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of sections (SectionID from file_0_view_0)
- $P$: set of products (ProductName from file_1_view_0)

Parameters:
- $v_p$: value (revenue) per unit of product $p$ (Value from file_1_view_0)
- $w_p$: space requirement per unit of product $p$ (Weight from file_1_view_0)
- $C_s$: total display space capacity of section $s$ (Capacity from file_0_view_0)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to stock in section $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \quad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

DATA MAPPING

Index Sets:
- $S$: file_0_view_0 SectionID
- $P$: file_1_view_0 ProductName

Parameters:
- $v_p$: file_1_view_0 Value, keyed by ProductName
- $w_p$: file_1_view_0 Weight, keyed by ProductName
- $C_s$: file_0_view_0 Capacity, keyed by SectionID

Decision Variables:
- $x_{sp}$: number of units of product $p$ in section $s$ (indexed by SectionID and ProductName)

All data is mapped directly from the returned rows and columns as specified above.