#### Index Sets
- Let $\mathcal{B}$ be the set of all products $i$ such that $\text{Product\_Name}_i$ contains "Books" in table_id "file_0_view_0".

#### Parameters
- $A_i$: Revenue per unit of product $i$, from column "Revenue" in table_id "file_0_view_0".
- $d_i$: Demand for product $i$, from column "Demand" in table_id "file_0_view_0".
- $I_i$: Initial inventory for product $i$, from column "Initial Inventory" in table_id "file_0_view_0".

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in \mathcal{B}$.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

#### Objective
$$
\max \sum_{i \in \mathcal{B}} A_i \cdot x_i
$$

#### Constraints
1. **Demand and Inventory Fulfillment Bounds:**
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in \mathcal{B}
   $$

#### Data Mapping
- All parameters ($A_i$, $d_i$, $I_i$) and index set $\mathcal{B}$ are defined using columns "Product_Name", "Revenue", "Demand", and "Initial Inventory" from table_id "file_0_view_0". Only records where "Product_Name" contains "Books" are included in $\mathcal{B}$.