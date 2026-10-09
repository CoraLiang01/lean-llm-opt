## Mathematical Model

### Sets
- $O$: Set of all generation contract options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ (coal, gas, renewables)
    - $gen_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $cost_o$: Cost per lot for option $o$ (from "cost_per_lot")

### Parameters
- $D = 200$: Total demand to be met

### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase of option $o \in O$

### Objective
Minimize total cost:
$$
\min \sum_{o \in O} cost_o \cdot x_o
$$

### Constraints

**Demand satisfaction:**
$$
\sum_{o \in O} gen_o \cdot x_o \geq D
$$

**Lot integrality:**
$$
x_o \in \mathbb{Z}_+, \quad \forall o \in O
$$

### Data Mapping

- $O$: All rows in energy.csv with tech in {coal, gas, renewables} (table_id: file_0_view_0, column: "option")
- $tech_o$: file_0_view_0, column: "tech"
- $gen_o$: file_0_view_0, column: "gen_per_lot"
- $cost_o$: file_0_view_0, column: "cost_per_lot"
- $D$: 200 (from user query)

**Summary:**  
Choose integer numbers of lots $x_o$ for each contract option $o$ to minimize total cost, such that the total generation meets or exceeds 200 units. All data is mapped directly from energy.csv as described.