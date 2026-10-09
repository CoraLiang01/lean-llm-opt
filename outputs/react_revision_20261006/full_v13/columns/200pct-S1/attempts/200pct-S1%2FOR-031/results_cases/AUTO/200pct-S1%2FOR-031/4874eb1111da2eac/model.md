## Mathematical Model

### Sets
- $O$: Set of all generation contract options (lots), indexed by $o$.
  - $O = \{$ all rows in energy.csv with columns: option, tech, gen_per_lot, cost_per_lot $\}$
- $T$: Set of generation technologies, $T = \{\text{coal}, \text{gas}, \text{renewables}\}$.

### Parameters (from Data Mapping)
- $g_o$: Generation per lot for option $o$ (from column gen_per_lot, table_id: file_0_view_0)
- $c_o$: Cost per lot for option $o$ (from column cost_per_lot, table_id: file_0_view_0)
- $D$: Total demand to be met, $D = 200$ (from user description)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o \, x_o
$$

### Constraints

**Demand Satisfaction**
$$
\sum_{o \in O} g_o \, x_o \geq D
$$

**Integrality**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

### Data Mapping

- $O$: All rows in energy.csv with columns [option, tech, gen_per_lot, cost_per_lot], table_id: file_0_view_0
- $g_o$: file_0_view_0, column gen_per_lot, indexed by option $o$
- $c_o$: file_0_view_0, column cost_per_lot, indexed by option $o$
- $D$: 200 (user description)
- $x_o$: integer variable for each $o \in O$

### Summary

- The model selects integer numbers of lots from each available contract option (coal, gas, renewables) to minimize total cost while meeting or exceeding the total demand of 200 units.
- All data for $g_o$ and $c_o$ is mapped directly from energy.csv, table_id: file_0_view_0, using the columns option, tech, gen_per_lot, cost_per_lot.
- No additional constraints or bounds are imposed unless specified in the data or question.