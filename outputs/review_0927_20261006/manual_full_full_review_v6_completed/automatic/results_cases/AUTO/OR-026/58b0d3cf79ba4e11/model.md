#### Index Sets
- $I$: Set of all products classified as ‘Fashion’ (from column ‘Product Name’ in table_id file_0_view_0).

#### Parameters
- $a_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’ in table_id file_0_view_0).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective
$$
\max \sum_{i \in I} a_i x_i
$$

#### Constraints
- Inventory and demand bounds:
  $$
  0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
  $$

#### Data Mapping
- All data is from table_id file_0_view_0 (SupermarketSales.csv), columns:
  - ‘Product Name’ (index set $I$)
  - ‘Revenue’ (parameter $a_i$)
  - ‘Demand’ (parameter $d_i$)
  - ‘Initial Inventory’ (parameter $s_i$)
- Filter: rows where ‘Product Name’ has prefix ‘Fashion’ (CSVQA predicate: ‘Product Name’ = ‘Fashion’ prefix).