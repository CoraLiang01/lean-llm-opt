## Mathematical Model

**Sets**
- $O$: set of all generation contract options (option), from table_id: file_0_view_0
- For each $o \in O$:
  - $c_o$: cost per lot (cost_per_lot)
  - $g_o$: generation per lot (gen_per_lot)

**Parameters**
- $D = 200$: total demand to be met

**Decision Variables**
- $x_o \in \mathbb{Z}_+, \quad \forall o \in O$: number of lots to purchase of option $o$ (must be integer and nonnegative)

**Objective**
$$
\min \sum_{o \in O} c_o\, x_o
$$

**Constraints**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

## Data Mapping

- $O$: All rows in file_0_view_0 (energy.csv), column "option"
- $c_o$: file_0_view_0, column "cost_per_lot"
- $g_o$: file_0_view_0, column "gen_per_lot"
- Demand $D$: 200 (from user description)
- $x_o$: integer, nonnegative, for each $o \in O$

**All generation options (coal, gas, renewables) are included. Each contract can be purchased in whole lots only.**