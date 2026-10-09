Mathematical Model

Sets:
- $I$: Set of all products with Product Name starting with "TABLET" (indexed by $i$).

Parameters:
- $r_i$: Revenue per unit of product $i$ (from column Revenue, table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column Demand, table_id file_0_view_0).
- $s_i$: Initial Inventory for product $i$ (from column Initial Inventory, table_id file_0_view_0).

Decision Variables:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
1. Inventory and Demand Fulfillment:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$

2. Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- $I$: All records in file_0_view_0 with Product Name starting with "TABLET".
- $r_i$: file_0_view_0, column Revenue, key Product Name.
- $d_i$: file_0_view_0, column Demand, key Product Name.
- $s_i$: file_0_view_0, column Initial Inventory, key Product Name.
- $x_i$: Decision variable for each Product Name in $I$.

All parameters and index sets are defined directly from the returned table file_0_view_0.