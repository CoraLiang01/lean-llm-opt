## Mathematical Model

### Sets
- $O$: Set of all contract options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ (coal, gas, renewables)
    - $c_o$: Cost per lot for option $o$ (energy.csv, "cost_per_lot")
    - $g_o$: Generation per lot for option $o$ (energy.csv, "gen_per_lot")

### Parameters
- $D$: Total demand to be met ($D = 200$)
- $c_o$: Cost per lot for option $o$ (energy.csv, "cost_per_lot")
- $g_o$: Generation per lot for option $o$ (energy.csv, "gen_per_lot")

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (integer, $x_o \geq 0$ and integer, for all $o \in O$)

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o \, x_o
$$

### Constraints

**Demand Satisfaction:**
$$
\sum_{o \in O} g_o \, x_o \geq D
$$

**Non-negativity and Integrality:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in energy.csv, column "option" (table_id: file_0_view_0, column: "option")
- $c_o$: energy.csv, column "cost_per_lot" (table_id: file_0_view_0, column: "cost_per_lot")
- $g_o$: energy.csv, column "gen_per_lot" (table_id: file_0_view_0, column: "gen_per_lot")
- $D$: 200 (from user description)
- $x_o$: Integer variable for each $o \in O$

All contract options in energy.csv are available for selection. Each $x_o$ is the number of lots (integer, $\geq 0$) of option $o$ to purchase. The sum of generation across all selected lots must meet or exceed the total demand $D=200$. The objective is to minimize the total procurement cost.