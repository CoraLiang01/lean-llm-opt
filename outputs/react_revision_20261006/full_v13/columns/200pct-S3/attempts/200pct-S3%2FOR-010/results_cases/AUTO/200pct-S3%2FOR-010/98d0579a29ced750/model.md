### Mathematical Model

Let:
- $S$ = set of section IDs from file_0_view_0[SectionID]
- $P$ = set of product names from file_1_view_0[ProductName]
- $x_{sp}$ = number of units of product $p \in P$ to be placed in section $s \in S$ (decision variable)

Parameters:
- $v_p$ = value of product $p$ (file_1_view_0[Value])
- $w_p$ = weight (space requirement) of product $p$ (file_1_view_0[Weight])
- $C_s$ = capacity of section $s$ (file_0_view_0[Capacity])

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

---

### Data Mapping

- Section index set $S$: file_0_view_0[SectionID]
- Product index set $P$: file_1_view_0[ProductName]
- Section capacity $C_s$: file_0_view_0[Capacity], keyed by SectionID
- Product value $v_p$: file_1_view_0[Value], keyed by ProductName
- Product weight $w_p$: file_1_view_0[Weight], keyed by ProductName
- Decision variables $x_{sp}$: integer, for all $s \in S$, $p \in P$