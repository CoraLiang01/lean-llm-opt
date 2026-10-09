##### Mathematical Model

Let $I$ be the set of all products in file_0_view_0 (Product Name) classified under ‘Fashion’.

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue column, file_0_view_0)
- $d_i$: Demand for product $i$ (from Demand column, file_0_view_0)
- $s_i$: Initial Inventory for product $i$ (from Initial Inventory column, file_0_view_0)

Decision variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
x_i \leq d_i \qquad \forall i \in I \\
x_i \leq s_i \qquad \forall i \in I \\
x_i \geq 0 \text{ and integer} \qquad \forall i \in I
$$

##### Data Mapping

- Index set $I$: All records in file_0_view_0, column Product Name, filtered as in the Observation.
- $r_i$: file_0_view_0, column Revenue, keyed by Product Name.
- $d_i$: file_0_view_0, column Demand, keyed by Product Name.
- $s_i$: file_0_view_0, column Initial Inventory, keyed by Product Name.
- Decision variable $x_i$: Number of units of product $i$ to fulfill, for each $i \in I$ (Product Name in file_0_view_0).