Mathematical Optimization Model

Index Sets:
Let $I$ be the set of all pizza types, as identified by the "Product Name" column.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of pizza type $i$ ("Revenue" column)
- $d_i$: Total demand for pizza type $i$ over the sales horizon ("Demand" column)
- $I_i$: Initial inventory available for pizza type $i$ ("Initial Inventory" column)

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \ x_i \geq 0$

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Constraints:
\[
x_i \leq d_i \qquad \forall i \in I
\]
\[
x_i \leq I_i \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_+, \ x_i \geq 0 \qquad \forall i \in I
\]

Data Mapping:
- Index set $I$ and all parameters $A_i$, $d_i$, $I_i$ are mapped from table_id: file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory" in PizzaSalesDataset.csv.