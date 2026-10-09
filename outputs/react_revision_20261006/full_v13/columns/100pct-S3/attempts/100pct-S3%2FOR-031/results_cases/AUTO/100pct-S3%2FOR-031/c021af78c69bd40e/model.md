## Mathematical Model

### Sets
- $O$: Set of all generation contract options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ ("coal", "gas", "renewables")
    - $g_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $c_o$: Cost per lot for option $o$ (from "cost_per_lot")

### Parameters
- $D = 200$: Total demand to be met

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o$ (integer, $x_o \geq 0$)

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

**Integrality and Non-negativity:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in energy.csv with "tech" in {"coal", "gas", "renewables"} (table_id: file_0_view_0, column: "option")
- $tech_o$: file_0_view_0, column: "tech"
- $g_o$: file_0_view_0, column: "gen_per_lot"
- $c_o$: file_0_view_0, column: "cost_per_lot"
- $D$: 200 (from user query)

**All variables are integer and nonnegative. Each $x_o$ represents the number of lots of contract $o$ to purchase.**