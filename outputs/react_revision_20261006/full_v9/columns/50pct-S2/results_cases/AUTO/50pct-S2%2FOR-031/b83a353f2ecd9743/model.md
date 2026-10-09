## Mathematical Model

### Sets
- $I$: Set of all generation options (indexed by $i$), from energy.csv, column "option".

### Parameters
- $g_i$: Generation per lot for option $i$ (energy.csv, column "gen_per_lot", table_id: file_0_view_0).
- $c_i$: Cost per lot for option $i$ (energy.csv, column "cost_per_lot", table_id: file_0_view_0).
- $D$: Total demand to be met, $D = 200$.

### Decision Variables
- $x_i \in \mathbb{Z}_+$: Number of lots to purchase for option $i$ (must be a non-negative integer).

### Objective
Minimize total cost:
$$
\min \sum_{i \in I} c_i\, x_i
$$

### Constraints
Meet total demand:
$$
\sum_{i \in I} g_i\, x_i \geq D
$$

Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All rows in energy.csv, table_id: file_0_view_0, column "option".
- $g_i$: energy.csv, table_id: file_0_view_0, column "gen_per_lot", for each $i$.
- $c_i$: energy.csv, table_id: file_0_view_0, column "cost_per_lot", for each $i$.
- $D$: 200 (from the question).

---

**Summary:**  
Choose integer numbers of lots $x_i$ for each generation option $i \in I$ to minimize total cost, such that the sum of generation meets or exceeds 200, using the per-lot generation and cost from the provided data.