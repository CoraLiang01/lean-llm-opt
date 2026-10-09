## Symbolic Mathematical Model

### Sets
- $O$: Set of all generation options (lots), indexed by $o$ (from table_id: file_0_view_0, column: option)
- $T$: Set of generation technologies, $T = \{\text{coal}, \text{gas}, \text{renewables}\}$

### Parameters (from data mapping)
- $g_o$: Generation per lot for option $o$ (file_0_view_0, column: gen_per_lot)
- $c_o$: Cost per lot for option $o$ (file_0_view_0, column: cost_per_lot)
- $t_o$: Technology type for option $o$ (file_0_view_0, column: tech)
- $D$: Total demand to meet ($D = 200$)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

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

**2. Integer Lot Purchases**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

### Data Mapping

- $O$: All rows in file_0_view_0, column "option"
- $g_o$: file_0_view_0, column "gen_per_lot"
- $c_o$: file_0_view_0, column "cost_per_lot"
- $t_o$: file_0_view_0, column "tech"
- $D$: 200 (from user description)

All variables $x_o$ are nonnegative integers, one for each lot option in the data. The model selects lots to minimize cost while meeting or exceeding the total demand.