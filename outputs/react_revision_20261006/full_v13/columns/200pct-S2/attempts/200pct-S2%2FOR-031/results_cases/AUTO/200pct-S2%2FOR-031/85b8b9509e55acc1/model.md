## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option"), indexed by $o$
- $T$: Set of generation technologies (coal, gas, renewables), indexed by $t$

### Parameters (from energy.csv, table_id: file_0_view_0)
- $c_o$: Cost per lot for option $o$ (column "cost_per_lot")
- $g_o$: Generation per lot for option $o$ (column "gen_per_lot")
- $tech_o$: Technology type for option $o$ (column "tech")
- $D$: Total demand to be met (scalar, $D = 200$)

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

**2. Integrality**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

### Data Mapping

- $O$ = all options in energy.csv with tech $\in$ {coal, gas, renewables} (table_id: file_0_view_0, column "option")
- $c_o$ = table_id: file_0_view_0, column "cost_per_lot", for each $o$
- $g_o$ = table_id: file_0_view_0, column "gen_per_lot", for each $o$
- $tech_o$ = table_id: file_0_view_0, column "tech", for each $o$
- $D = 200$ (from user description)

All variables $x_o$ are nonnegative integers (whole lots), and all options in the data are available for selection. The model minimizes total procurement cost while ensuring total generation meets or exceeds demand.