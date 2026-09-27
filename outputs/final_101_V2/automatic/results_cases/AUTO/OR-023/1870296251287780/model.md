#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘ELE-S’ (identified by Product_Reference in the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column ‘Revenue’)
    - $d_i$ = deterministic demand for product $i$ (parameter from column ‘Demand’)
    - $s_i$ = initial inventory for product $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Demand fulfillment constraint:
  $$
  x_i \leq d_i \quad \forall i \in I
  $$
- Inventory constraint:
  $$
  x_i \leq s_i \quad \forall i \in I
  $$
- Non-negativity and integrality:
  $$
  x_i \in \mathbb{Z}_{+} \quad \forall i \in I
  $$

#### Data Mapping

- Table: SalesStoreoverview.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Product_Reference’ starts with ‘ELE-S’
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $s_i$: column ‘Initial Inventory’