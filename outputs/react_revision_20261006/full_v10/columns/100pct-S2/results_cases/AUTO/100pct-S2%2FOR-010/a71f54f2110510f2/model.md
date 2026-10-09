ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of sections (SectionID from file_0_view_0)
- $P$: set of products (ProductName from file_1_view_0)

Parameters:
- $c_s$: display space capacity of section $s$ (Capacity from file_0_view_0, indexed by SectionID)
- $v_p$: value (price) of product $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: shelf space requirement of product $p$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

DATA MAPPING

Index Sets:
- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)

Parameters:
- $c_s$: file_0_view_0, column Capacity, key SectionID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

Decision Variables:
- $x_{sp}$: number of units of product $p$ to stock in section $s$ (integer, nonnegative)