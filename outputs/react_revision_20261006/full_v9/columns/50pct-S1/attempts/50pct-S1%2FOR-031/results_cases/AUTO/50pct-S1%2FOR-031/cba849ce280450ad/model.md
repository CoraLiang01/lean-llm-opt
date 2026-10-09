## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ (coal, gas, renewables)
    - $g_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $c_o$: Cost per lot for option $o$ (from "cost_per_lot")

### Parameters
- $D$: Total demand to be met, $D = 200$

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

**Demand Satisfaction**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

**Lot Integrality**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in energy.csv, column "option" (table_id: file_0_view_0, column: option)
- $tech_o$: energy.csv, column "tech" (table_id: file_0_view_0, column: tech)
- $g_o$: energy.csv, column "gen_per_lot" (table_id: file_0_view_0, column: gen_per_lot)
- $c_o$: energy.csv, column "cost_per_lot" (table_id: file_0_view_0, column: cost_per_lot)
- $D$: 200 (from user query)

- $x_o$: Integer, nonnegative, for each $o \in O$

**All options in energy.csv are available for selection.**