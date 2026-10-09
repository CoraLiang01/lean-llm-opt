## Mathematical Model

### Sets
- $O$: Set of all generation options (lots), indexed by $o$ (from table_id: file_0_view_0, column: option)
- For each $o \in O$, let $\text{tech}_o$ denote its technology type (coal, gas, renewables)

### Parameters (from data mapping)
- $g_o$: Generation per lot for option $o$ (file_0_view_0, column: gen_per_lot)
- $c_o$: Cost per lot for option $o$ (file_0_view_0, column: cost_per_lot)
- $D$: Total demand to meet (given in question, $D = 200$)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (must be integer and $\geq 0$)

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

**1. Demand Satisfaction**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

**2. Integer and Non-negativity**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

### Data Mapping

- $O$: All rows in file_0_view_0, column "option"
- $g_o$: file_0_view_0, column "gen_per_lot", for each $o$
- $c_o$: file_0_view_0, column "cost_per_lot", for each $o$
- $x_o$: integer variable for each $o \in O$
- $D$: 200 (from question)

### Summary

- Each $x_o$ is the integer number of lots of generation option $o$ to purchase.
- The sum of all purchased generation must meet or exceed 200.
- The objective is to minimize total cost, summing over all options.
- All data is mapped directly from file_0_view_0 (energy.csv) as described above.