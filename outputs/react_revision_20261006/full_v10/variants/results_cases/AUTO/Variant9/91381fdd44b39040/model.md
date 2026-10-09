Mathematical Model

Index Sets:
- Let I be the set of item types (from file_0_view_0: Item).
- Let P be the set of cutting patterns (from file_1_view_0: Pattern).

Parameters:
- $d_i$: demand for item $i \in I$ (file_0_view_0: Demand, indexed by Item).
- $a_{ip}$: number of units of item $i$ produced by one standard roll cut with pattern $p$ (file_1_view_0: column $i$, row Pattern $p$).

Decision Variables:
- $y_p \in \mathbb{Z}_{\geq 0}$: number of standard rolls cut using pattern $p \in P$.

Objective:
Minimize the total number of standard rolls used:
$$
\min \sum_{p \in P} y_p
$$

Constraints:
1. Demand satisfaction for each item type:
$$
\sum_{p \in P} a_{ip} y_p \geq d_i \quad \forall i \in I
$$

2. Nonnegativity and integrality:
$$
y_p \in \mathbb{Z}_{\geq 0} \quad \forall p \in P
$$

Data Mapping

Index Sets:
- $I$: file_0_view_0, column Item
- $P$: file_1_view_0, column Pattern

Parameters:
- $d_i$: file_0_view_0, column Demand, indexed by Item $i$
- $a_{ip}$: file_1_view_0, column $i$ (Item), row Pattern $p$

Variables:
- $y_p$: integer, nonnegative, for each Pattern $p$ (file_1_view_0, column Pattern)

Objective:
- Minimize $\sum_{p \in P} y_p$

Constraints:
- For each $i \in I$: $\sum_{p \in P} a_{ip} y_p \geq d_i$
- For each $p \in P$: $y_p \in \mathbb{Z}_{\geq 0}$