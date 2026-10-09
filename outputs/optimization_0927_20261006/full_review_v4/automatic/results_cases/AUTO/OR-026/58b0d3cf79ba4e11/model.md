#### Index Sets

- $I$: Set of all products classified as ‘Fashion’ (from column ‘Product Name’ in table_id file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Bounds:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

2. Variable Domain:
   $$
   x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

#### Data Mapping

- All data is from table_id file_0_view_0 (SupermarketSales.csv).
- Index set $I$ is defined by all rows where ‘Product Name’ has prefix ‘Fashion’ (filter: ‘Product Name’ = ‘Fashion’ prefix).
- Parameters $A_i$, $d_i$, $I_i$ are mapped to columns ‘Revenue’, ‘Demand’, and ‘Initial Inventory’ respectively, for each $i \in I$.