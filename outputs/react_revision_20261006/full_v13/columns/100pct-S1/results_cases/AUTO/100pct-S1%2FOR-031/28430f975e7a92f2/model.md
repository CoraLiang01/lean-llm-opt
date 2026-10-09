## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option"), indexed by $o$.
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ ("coal", "gas", "renewables").
    - $gen_o$: Generation per lot for option $o$ (from "gen_per_lot", units consistent with demand).
    - $cost_o$: Cost per lot for option $o$ (from "cost_per_lot").

### Parameters
- $D$: Total demand to be met, $D = 200$.

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (must be integer and nonnegative).

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} cost_o \cdot x_o
$$

### Constraints

**Demand Satisfaction:**
$$
\sum_{o \in O} gen_o \cdot x_o \geq D
$$

**Integrality and Nonnegativity:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All 131 rows in energy.csv, column "option" (table_id: file_0_view_0, column: "option").
- $tech_o$: energy.csv, column "tech" (table_id: file_0_view_0, column: "tech").
- $gen_o$: energy.csv, column "gen_per_lot" (table_id: file_0_view_0, column: "gen_per_lot").
- $cost_o$: energy.csv, column "cost_per_lot" (table_id: file_0_view_0, column: "cost_per_lot").
- $D$: 200 (from user description).

- $x_o$: Integer variable for each $o \in O$.

---

**Summary:**  
Choose integer numbers of lots $x_o$ for each generation option $o$ to minimize total cost, such that the total generation meets or exceeds 200 units. All data is mapped directly from energy.csv as described.