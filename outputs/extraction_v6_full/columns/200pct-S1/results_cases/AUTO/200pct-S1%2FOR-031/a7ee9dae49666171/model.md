## Abstract Mathematical Model

Let:
- $O$ = set of generation options (from energy.csv, column option)
- For each $o \in O$:
    - $c_o$ = cost per lot (energy.csv, column cost_per_lot)
    - $g_o$ = generation per lot (energy.csv, column gen_per_lot)
    - $x_o$ = number of lots of option $o$ to purchase (decision variable, integer, $x_o \geq 0$)

Parameters:
- $D$ = total demand to meet (given as 200)

### Decision Variables
- $x_o \in \mathbb{Z}_{\geq 0}$, for all $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o \, x_o
$$

### Constraints
Meet or exceed total demand:
$$
\sum_{o \in O} g_o \, x_o \geq D
$$

Nonnegativity and integrality:
$$
x_o \in \mathbb{Z}_{\geq 0} \quad \forall o \in O
$$

---

## Data Mapping

- $O$ (generation options): file_0_view_0.option
- $c_o$ (cost per lot): file_0_view_0.cost_per_lot, indexed by file_0_view_0.option
- $g_o$ (generation per lot): file_0_view_0.gen_per_lot, indexed by file_0_view_0.option
- $x_o$ (decision variable): indexed by file_0_view_0.option
- $D$ (demand): 200 (from user query)

All parameters are to be used exactly as returned, preserving file and column names for mapping.