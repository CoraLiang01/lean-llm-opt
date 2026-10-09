#### Symbolic Mathematical Model

Let:
- $I$ = index set of all pizza types (from the dataset)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (parameter, from column "Revenue")
    - $d_i$ = total demand for pizza type $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory for pizza type $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory and demand fulfillment constraints:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
$$
or equivalently,
$$
x_i \leq d_i \qquad \forall i \in I
$$
$$
x_i \leq s_i \qquad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (PizzaSalesDataset.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $s_i$: Column "Initial Inventory"
    - Decision variable $x_i$: Number of fulfilled units for each $i \in I$