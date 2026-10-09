## Mathematical Model

### Sets
- $O$: Set of all generation options (from energy.csv, column "option"), indexed by $o$
- $T$: Set of generation types (coal, gas, renewables), $T = \{\text{coal}, \text{gas}, \text{renewables}\}$
- For each $o \in O$, let $\text{tech}_o$ be its technology type (from "tech")

### Parameters (from Data Mapping)
- $c_o$: Cost per lot for option $o$ (energy.csv, column "cost_per_lot", table_id: file_0_view_0)
- $g_o$: Generation per lot for option $o$ (energy.csv, column "gen_per_lot", table_id: file_0_view_0)
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

- $O$: All rows in energy.csv with "tech" in $\{\text{coal}, \text{gas}, \text{renewables}\}$ (table_id: file_0_view_0, column "option")
- $c_o$: energy.csv, column "cost_per_lot", table_id: file_0_view_0, for each $o$
- $g_o$: energy.csv, column "gen_per_lot", table_id: file_0_view_0, for each $o$
- $x_o$: integer variable for each $o \in O$
- $D$: 200 (from user description)

### Summary

- Minimize total cost of purchased lots.
- Meet or exceed total demand of 200 units.
- Each option can be purchased in integer lots only.
- All data for $O$, $c_o$, $g_o$ from energy.csv, table_id: file_0_view_0, columns as above.