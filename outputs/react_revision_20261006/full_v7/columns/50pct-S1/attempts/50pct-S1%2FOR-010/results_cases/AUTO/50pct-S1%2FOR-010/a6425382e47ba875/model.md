#### Mathematical Model

Let:
- $S$ = set of section IDs from file_0_view_0, column SectionID
- $P$ = set of product IDs from file_1_view_0, column ProductName

Parameters:
- $c_s$ = capacity of section $s$ (from file_0_view_0, column Capacity, keyed by SectionID)
- $v_p$ = value (price) of product $p$ (from file_1_view_0, column Value, keyed by ProductName)
- $w_p$ = space requirement (weight) of product $p$ (from file_1_view_0, column Weight, keyed by ProductName)

Decision variables:
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

---

#### Data Mapping

- $S$: file_0_view_0, column SectionID
- $P$: file_1_view_0, column ProductName
- $c_s$: file_0_view_0, columns SectionID (key), Capacity (value)
- $v_p$: file_1_view_0, columns ProductName (key), Value (value)
- $w_p$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{sp}$: decision variable for each $(s,p) \in S \times P$