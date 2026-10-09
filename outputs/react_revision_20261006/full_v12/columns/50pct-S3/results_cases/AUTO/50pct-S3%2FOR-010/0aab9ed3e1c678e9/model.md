### Mathematical Model

Let:
- $S$ = set of sections (indexed by $s$), with SectionID from file_0_view_0
- $P$ = set of products (indexed by $p$), with ProductName from file_1_view_0

Parameters:
- $c_s$ = capacity of section $s$ (Capacity in file_0_view_0)
- $v_p$ = value of product $p$ (Value in file_1_view_0)
- $w_p$ = space requirement of product $p$ (Weight in file_1_view_0)

Decision variables:
- $x_{sp}$ = number of units of product $p$ to stock in section $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

---

### Data Mapping

- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $c_s$: Capacity column in file_0_view_0, keyed by SectionID
- $v_p$: Value column in file_1_view_0, keyed by ProductName
- $w_p$: Weight column in file_1_view_0, keyed by ProductName
- $x_{sp}$: Number of units of product $p$ in section $s$ (decision variable, integer, nonnegative)