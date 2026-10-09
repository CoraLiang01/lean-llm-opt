ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $S$ be the set of SectionIDs from file_0_view_0 (capacity.csv).
- Let $P$ be the set of ProductNames from file_1_view_0 (products.csv).

Parameters:
- $c_s$: Capacity of section $s \in S$, from file_0_view_0, column Capacity.
- $v_p$: Value (price) of product $p \in P$, from file_1_view_0, column Value.
- $w_p$: Shelf space requirement (Weight) of product $p \in P$, from file_1_view_0, column Weight.

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to stock in section $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$.

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
- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName

Parameters:
- $c_s$: file_0_view_0, column Capacity, keyed by SectionID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to stock in section $s$ (integer, $\geq 0$)