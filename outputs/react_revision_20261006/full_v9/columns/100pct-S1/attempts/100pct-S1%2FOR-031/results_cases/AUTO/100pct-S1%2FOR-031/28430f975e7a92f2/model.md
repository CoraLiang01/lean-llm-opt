## Mathematical Model

### Sets
- $O$: Set of all generation options (lots), indexed by $o$ (from all rows in energy.csv with tech $\in$ {coal, gas, renewables})
- For each $o \in O$:
  - $option_o$: unique identifier (energy.csv: option)
  - $tech_o$: technology type (energy.csv: tech)
  - $g_o$: generation per lot (energy.csv: gen_per_lot)
  - $c_o$: cost per lot (energy.csv: cost_per_lot)

### Parameters
- $D$: Total demand to meet (given as $200$)

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} c_o\, x_o
$$

### Constraints

**Demand Satisfaction**
$$
\sum_{o \in O} g_o\, x_o \geq D
$$

**Lot Integrality**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

---

### Data Mapping

- $O$: All rows in table_id: file_0_view_0, columns: option, tech, gen_per_lot, cost_per_lot, with tech $\in$ {coal, gas, renewables}
- $g_o$: file_0_view_0.gen_per_lot for option $o$
- $c_o$: file_0_view_0.cost_per_lot for option $o$
- $D$: 200 (from user description)
- $x_o$: integer variable for each $o \in O$

---

**Summary:**  
Minimize total cost of purchased lots, choosing integer numbers of lots from any available coal, gas, or renewables option, so that the total generation meets or exceeds 200. Each lot's generation and cost are as listed in energy.csv. All $x_o$ are nonnegative integers.