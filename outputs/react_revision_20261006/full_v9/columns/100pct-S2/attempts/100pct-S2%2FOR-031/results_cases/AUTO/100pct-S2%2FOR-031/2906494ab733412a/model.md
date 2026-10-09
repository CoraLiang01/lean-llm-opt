## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ (coal, gas, renewables)
    - $g_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $c_o$: Cost per lot for option $o$ (from "cost_per_lot")

### Parameters
- $D = 200$: Total demand to be met

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

**Demand satisfaction:**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

**Lot integrality:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in energy.csv with "tech" in $\{\text{coal}, \text{gas}, \text{renewables}\}$ (table_id: file_0_view_0, column: "option")
- $tech_o$: energy.csv, column "tech", for each $o$
- $g_o$: energy.csv, column "gen_per_lot", for each $o$
- $c_o$: energy.csv, column "cost_per_lot", for each $o$
- $D$: 200 (from user description)
- $x_o$: integer variable for each $o \in O$

**All generation options and their parameters are taken directly from energy.csv, table_id file_0_view_0, columns "option", "tech", "gen_per_lot", "cost_per_lot".**