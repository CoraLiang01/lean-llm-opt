#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘ELE-S’ (from column Product_Reference)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column Revenue)
    - $d_i$ = deterministic demand for product $i$ (from column Demand)
    - $s_i$ = initial inventory for product $i$ (from column Initial Inventory)
    - $x_i$ = integer decision variable: number of units of product $i$ to fulfill

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory constraints:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand constraints:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- Table: SalesStoreoverview.csv (table_id: file_0_view_0)
    - Index set $I$: all rows where Product_Reference starts with ‘ELE-S’
    - $A_i$: column Revenue
    - $d_i$: column Demand
    - $s_i$: column Initial Inventory
    - $i$: column Product_Reference

No additional constraints or parameters are imposed by the query. All data columns and index sets are mapped directly from the specified table and columns.