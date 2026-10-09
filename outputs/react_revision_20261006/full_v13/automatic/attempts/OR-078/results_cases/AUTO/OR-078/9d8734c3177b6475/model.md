## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option"), indexed by $o$.

### Parameters (from Data Mapping)
- $g_o$: Generation per lot for option $o$ (energy.csv, column "gen_per_lot", table_id: file_0_view_0)
- $c_o$: Cost per lot for option $o$ (energy.csv, column "cost_per_lot", table_id: file_0_view_0)
- $D$: Total demand to be met (given: $D = 200$)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (must be integer and nonnegative)

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

**Demand Satisfaction:**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

**Integrality and Nonnegativity:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in energy.csv, column "option", table_id: file_0_view_0
- $g_o$: energy.csv, column "gen_per_lot", table_id: file_0_view_0
- $c_o$: energy.csv, column "cost_per_lot", table_id: file_0_view_0
- $D$: 200 (from user description)

---

**Summary:**  
Choose integer numbers of lots $x_o$ for each generation option $o \in O$ to minimize total cost, such that the total generation meets or exceeds 200 units. Each $x_o$ is integer and nonnegative. All parameters are mapped directly from the provided energy.csv.