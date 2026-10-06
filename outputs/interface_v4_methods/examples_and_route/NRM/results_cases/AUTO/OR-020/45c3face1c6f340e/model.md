#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business identifier Product Name from SalesDatainBusinesses.csv)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (parameter: Revenue)
    - $d_i$ = demand for product $i$ (parameter: Demand)
    - $s_i$ = initial inventory of product $i$ (parameter: Initial Inventory)
- Decision variable: $x_i$ = number of units of product $i$ to fulfill

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

#### Data Mapping

- $I$: Product Name (SalesDatainBusinesses.csv)
- $r_i$: Revenue (SalesDatainBusinesses.csv, column: Revenue, table_id: file_0_view_0)
- $d_i$: Demand (SalesDatainBusinesses.csv, column: Demand, table_id: file_0_view_0)
- $s_i$: Initial Inventory (SalesDatainBusinesses.csv, column: Initial Inventory, table_id: file_0_view_0)
- $x_i$: Decision variable for product $i$ (indexed by Product Name)

All parameters and variables are indexed by the original Product Name. All 109 products and their associated data are included, in the original file and row order.