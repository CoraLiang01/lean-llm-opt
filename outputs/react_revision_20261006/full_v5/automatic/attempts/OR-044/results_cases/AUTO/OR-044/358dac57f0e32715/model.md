Mathematical Optimization Model

Index Sets:
- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

Parameters:
- $c_s$: capacity of section $s$ (from file_0_view_0, column Capacity, indexed by SectionID)
- $v_p$: value (price) of product $p$ (from file_1_view_0, column Value, indexed by ProductName)
- $w_p$: shelf space requirement of product $p$ (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to stock in section $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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
- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName
- $c_s$: file_0_view_0, columns SectionID and Capacity
- $v_p$: file_1_view_0, columns ProductName and Value
- $w_p$: file_1_view_0, columns ProductName and Weight
- $x_{sp}$: decision variable for each $(s,p) \in S \times P$