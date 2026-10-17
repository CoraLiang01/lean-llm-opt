#### Index Sets

- $I$: Set of all products classified as ‘FAUX’ (from column ‘Product Name’ in table_id: file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraints:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. Demand constraints:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) and index set $I$ are drawn from table_id: file_0_view_0 in ZARASales.csv, using only rows where ‘Product Name’ has prefix ‘FAUX’, and columns ‘Product Name’, ‘Revenue’, ‘Demand’, and ‘Initial Inventory’.