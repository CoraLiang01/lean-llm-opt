ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of sections, with elements $s$ corresponding to SectionID from file_0_view_0 (capacity.csv)
- $P$: set of products, with elements $p$ corresponding to ProductName from file_1_view_0 (products.csv)

Parameters:
- $C_s$: display space capacity of section $s$ (Capacity column, file_0_view_0)
- $v_p$: value (price) of product $p$ (Value column, file_1_view_0)
- $w_p$: shelf space requirement of product $p$ (Weight column, file_1_view_0)

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

- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $C_s$: file_0_view_0, column Capacity, key SectionID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $x_{sp}$: number of units of product $p$ to stock in section $s$ (decision variable, indexed by SectionID and ProductName)