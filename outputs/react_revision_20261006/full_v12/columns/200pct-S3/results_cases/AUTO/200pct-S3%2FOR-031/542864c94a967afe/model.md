## Mathematical Model

**Sets**
- $O$: set of all generation options (option IDs from energy.csv, table_id: file_0_view_0)
- For each $o \in O$:
  - $tech_o$: technology type of option $o$ (coal, gas, renewables)
  - $gen_o$: generation per lot for option $o$ (from gen_per_lot)
  - $cost_o$: cost per lot for option $o$ (from cost_per_lot)

**Parameters**
- $D = 200$: total demand to be met

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

**Objective**
\[
\min \sum_{o \in O} cost_o \cdot x_o
\]

**Constraints**
\[
\sum_{o \in O} gen_o \cdot x_o \geq D
\]
\[
x_o \in \mathbb{Z}_+, \quad \forall o \in O
\]

**Data Mapping**
- $O$: All rows in energy.csv with tech in $\{\text{coal}, \text{gas}, \text{renewables}\}$ (table_id: file_0_view_0, column: option)
- $tech_o$: table_id: file_0_view_0, column: tech
- $gen_o$: table_id: file_0_view_0, column: gen_per_lot
- $cost_o$: table_id: file_0_view_0, column: cost_per_lot
- $x_o$: integer variable for each $o \in O$
- $D$: demand, given as 200

**Summary**
- Choose integer numbers of lots $x_o$ for each available option $o$ to minimize total cost, such that total generation meets or exceeds 200 units. Each $x_o$ is nonnegative integer. All data is mapped directly from energy.csv as described.