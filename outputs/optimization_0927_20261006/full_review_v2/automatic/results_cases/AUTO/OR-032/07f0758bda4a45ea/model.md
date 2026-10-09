#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified as ‘Books’ (from Product_Name).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from Revenue).
    - $I_i$: Initial inventory of product $i$ (from Initial Inventory).
    - $d_i$: Demand for product $i$ (from Demand).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

##### Variables

$$
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
$$

##### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

##### Constraints

1. Inventory constraint:
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$

##### Data Mapping

- Table: file_0_view_0 (DifferentStoreSales.csv)
    - Index set $\mathcal{I}$: All rows where Product_Name has prefix ‘Books_’
    - $A_i$: Revenue (column: Revenue)
    - $I_i$: Initial Inventory (column: Initial Inventory)
    - $d_i$: Demand (column: Demand)
    - $i$: Product_Name (column: Product_Name)