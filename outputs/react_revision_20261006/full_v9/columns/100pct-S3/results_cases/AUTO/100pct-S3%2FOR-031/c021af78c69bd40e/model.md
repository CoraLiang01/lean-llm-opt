## Mathematical Model

### Sets
- $O$: Set of all available generation contract options (lots), indexed by $o$.
  - Data Mapping: $O = \{$ all rows in file_0_view_0, column "option" $\}$
- For each $o \in O$:
  - $g_o$: generation technology of option $o$ (coal, gas, renewables)
  - $G_o$: generation per lot for option $o$ (from "gen_per_lot")
  - $C_o$: cost per lot for option $o$ (from "cost_per_lot")

### Parameters
- $D$: Total demand to be met. $D = 200$ (given)
- $G_o$: Generation per lot for option $o$ (file_0_view_0, column "gen_per_lot")
- $C_o$: Cost per lot for option $o$ (file_0_view_0, column "cost_per_lot")

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (integer, $x_o \geq 0$), for all $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} C_o \, x_o
$$

### Constraints

**1. Demand Satisfaction**
$$
\sum_{o \in O} G_o \, x_o \geq D
$$

**2. Integer and Non-negativity**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: file_0_view_0, column "option"
- $g_o$: file_0_view_0, column "tech"
- $G_o$: file_0_view_0, column "gen_per_lot"
- $C_o$: file_0_view_0, column "cost_per_lot"
- $D$: 200 (from user description)
- $x_o$: integer variable, number of lots of option $o$

---

**Summary:**  
Choose integer numbers of lots $x_o$ for each contract option $o$ (coal, gas, renewables) to minimize total cost, such that the total generation meets or exceeds 200, using the per-lot generation and cost data from file_0_view_0 (energy.csv). Each $x_o$ is integer and nonnegative.