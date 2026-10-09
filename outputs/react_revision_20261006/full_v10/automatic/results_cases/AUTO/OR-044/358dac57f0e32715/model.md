ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $S$ be the set of SectionIDs from file_0_view_0 (capacity.csv).
- Let $P$ be the set of ProductNames from file_1_view_0 (products.csv).

Parameters:
- $c_s$: Capacity of section $s \in S$, from file_0_view_0, column Capacity.
- $v_p$: Value (price) of product $p \in P$, from file_1_view_0, column Value.
- $w_p$: Weight (space requirement) of product $p \in P$, from file_1_view_0, column Weight.

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ to be placed in section $s$.

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Constraints:
1. Section capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s
\]

2. Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
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
- $x_{sp}$: Number of units of product $p$ in section $s$ (integer, $\geq 0$)