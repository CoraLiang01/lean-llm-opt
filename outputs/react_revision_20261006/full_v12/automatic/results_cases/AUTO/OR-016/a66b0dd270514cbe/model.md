#### Mathematical Optimization Model

Let:
- $I$ = set of all products (indexed by $i$), as defined by all "Product Name" entries in table_id file_0_view_0.

Parameters (for each $i \in I$):
- $A_i$ = revenue per unit of product $i$ (from column "Revenue")
- $d_i$ = demand for product $i$ (from column "Demand")
- $s_i$ = initial inventory for product $i$ (from column "Initial Inventory")

Decision variables:
- $x_i$ = number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory limits:
$$
x_i \leq s_i \quad \forall i \in I
$$

- Demand limits:
$$
x_i \leq d_i \quad \forall i \in I
$$

- Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

#### Data Mapping

- Index set $I$: All "Product Name" values in table_id file_0_view_0 (RetailSalesDataset.csv)
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $s_i$: "Initial Inventory" column in table_id file_0_view_0
- Decision variable $x_i$: fulfillment quantity for each $i \in I$