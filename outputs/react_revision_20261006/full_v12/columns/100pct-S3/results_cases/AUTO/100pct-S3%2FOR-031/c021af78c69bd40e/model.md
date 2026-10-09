## Mathematical Model

**Sets**
- $O$: set of all generation options (option IDs in energy.csv, table_id: file_0_view_0)
- For each $o \in O$:
    - $tech_o$: technology type of option $o$ (coal, gas, renewables)
    - $gen\_per\_lot_o$: generation per lot for option $o$
    - $cost\_per\_lot_o$: cost per lot for option $o$

**Parameters**
- $D = 200$: total demand to be met

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

**Objective**
\[
\min \sum_{o \in O} cost\_per\_lot_o \cdot x_o
\]

**Constraints**
\[
\sum_{o \in O} gen\_per\_lot_o \cdot x_o \geq D
\]
\[
x_o \in \mathbb{Z}_+, \quad \forall o \in O
\]

**Data Mapping**
- $O$: All rows in energy.csv with $tech$ in $\{\text{coal}, \text{gas}, \text{renewables}\}$ (table_id: file_0_view_0, column: option)
- $gen\_per\_lot_o$: column gen_per_lot, table_id: file_0_view_0
- $cost\_per\_lot_o$: column cost_per_lot, table_id: file_0_view_0
- $x_o$: integer variable for each $o \in O$
- $D$: scalar, 200 (from user description)

**Summary**
- Choose integer lots $x_o$ for each generation option $o$ to minimize total cost, such that total generation meets or exceeds 200 units. All data is mapped directly from energy.csv, table_id: file_0_view_0.