#### Mathematical Optimization Model

Let:
- $S$ = set of sections, indexed by $s$, with SectionID from file_0_view_0.
- $P$ = set of products, indexed by $p$, with ProductName from file_1_view_0.

Parameters:
- $c_s$ = capacity of section $s$ (from file_0_view_0, column Capacity).
- $v_p$ = value (price) of product $p$ (from file_1_view_0, column Value).
- $w_p$ = space requirement (weight) of product $p$ (from file_1_view_0, column Weight).

Decision variables:
- $x_{sp}$ = number of units of product $p$ to stock in section $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$.

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

---

#### Data Mapping

- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $c_s$: file_0_view_0, column Capacity, keyed by SectionID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{sp}$: number of units of product $p$ in section $s$ (decision variable, integer, nonnegative)