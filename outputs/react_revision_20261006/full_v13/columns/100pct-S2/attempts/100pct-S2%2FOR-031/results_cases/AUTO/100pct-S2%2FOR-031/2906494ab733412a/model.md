## Mathematical Model

### Sets
- $O$: Set of all generation options (option IDs) from energy.csv, $O = \{\text{coal\_001}, \ldots, \text{renewables\_056}\}$
- For each $o \in O$, let $\text{tech}_o \in \{\text{coal}, \text{gas}, \text{renewables}\}$

### Parameters (from energy.csv, table_id: file_0_view_0)
- $g_o$: Generation per lot for option $o$ (column: gen_per_lot)
- $c_o$: Cost per lot for option $o$ (column: cost_per_lot)
- $D$: Total demand to meet (scalar, $D = 200$)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

1. **Demand Satisfaction**
   $$
   \sum_{o \in O} g_o\, x_o \geq D
   $$

2. **Lot Integrality and Nonnegativity**
   $$
   x_o \in \mathbb{Z}_+, \quad \forall o \in O
   $$

### Data Mapping

- $O$: All records in energy.csv with columns: option, tech, gen_per_lot, cost_per_lot (table_id: file_0_view_0)
- $g_o$: file_0_view_0, column "gen_per_lot", for each $o$
- $c_o$: file_0_view_0, column "cost_per_lot", for each $o$
- $x_o$: integer variable for each $o \in O$
- $D$: scalar, 200 (from user description)

### Summary

- Each $x_o$ is the integer number of lots of generation option $o$ to purchase.
- The sum of all generation must meet or exceed 200.
- The objective is to minimize total cost.
- All data is mapped directly from energy.csv, table_id: file_0_view_0, using columns "option", "tech", "gen_per_lot", "cost_per_lot".