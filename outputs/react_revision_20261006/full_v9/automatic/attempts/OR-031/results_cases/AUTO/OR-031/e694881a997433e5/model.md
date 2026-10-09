#### Symbolic Mathematical Model

Let:
- $I$ = set of all dairy products (indexed by $i$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = demand for product $i$
    - $s_i$ = initial inventory for product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
1. Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

#### Data Mapping

- Table: file_0_view_0 (DairyGoodsSalesDataset.csv)
    - Index set $I$: All unique values in column Full_Product_Name
    - Parameter $A_i$: Revenue from column Revenue
    - Parameter $d_i$: Demand from column Demand
    - Parameter $s_i$: Initial Inventory from column Initial Inventory