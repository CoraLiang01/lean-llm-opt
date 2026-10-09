### Mathematical Model

Let:
- $S$ = set of sections (indexed by $s$), from file_0_view_0:SectionID
- $P$ = set of products (indexed by $p$), from file_1_view_0:ProductName

Parameters:
- $C_s$ = capacity of section $s$, from file_0_view_0:Capacity
- $v_p$ = value of product $p$, from file_1_view_0:Value
- $w_p$ = space requirement (weight) of product $p$, from file_1_view_0:Weight

Decision variables:
- $x_{sp}$ = number of units of product $p$ to stock in section $s$, integer and $\geq 0$

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

Subject to:
$$
\sum_{p \in P} w_p \, x_{sp} \leq C_s \quad \forall s \in S
$$

$$
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

---

### Data Mapping

- $S$: file_0_view_0:SectionID
- $P$: file_1_view_0:ProductName
- $C_s$: file_0_view_0:Capacity (keyed by SectionID)
- $v_p$: file_1_view_0:Value (keyed by ProductName)
- $w_p$: file_1_view_0:Weight (keyed by ProductName)
- $x_{sp}$: number of units of product $p$ in section $s$ (decision variable, integer, $\geq 0$)