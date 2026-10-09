## Mathematical Model

Let:
- $I$ = set of items (indexed by $i$)
- $P$ = set of cutting patterns (indexed by $p$)
- $d_i$ = demand for item $i$
- $a_{ip}$ = number of pieces of item $i$ produced by pattern $p$
- $y_p$ = number of times pattern $p$ is used (decision variable)

### Objective
Minimize the total number of rolls used:
$$
\min \sum_{p \in P} y_p
$$

### Constraints

1. **Demand Satisfaction** (for each item $i$):
$$
\sum_{p \in P} a_{ip} y_p \geq d_i \quad \forall i \in I
$$

2. **Nonnegativity and Integrality** (for each pattern $p$):
$$
y_p \in \mathbb{Z}_{\geq 0} \quad \forall p \in P
$$

---

## Data Mapping

- $I$: All Item values from file_0_view_0 column "Item"
- $P$: All Pattern values from file_1_view_0 column "Pattern"
- $d_i$: file_0_view_0, column "Demand", keyed by "Item"
- $a_{ip}$: file_1_view_0, column for item $i$, row for pattern $p$ (keyed by "Pattern" and "Item")
- $y_p$: Decision variable for each pattern $p \in P$