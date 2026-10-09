#### Index Sets

- $I$: Set of all products with names starting with "S700_" (from column "Product Name" in table_id file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: Total demand for product $i$ (from column "Demand" in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, for all $i \in I$.

#### Objective

$\max \sum_{i \in I} A_i \cdot x_i$

#### Constraints

1. Inventory constraint:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

#### Data Mapping

- Index set $I$ is defined by all records in table_id file_0_view_0 where "Product Name" has prefix "S700_".
- Parameter $A_i$ is mapped from column "Revenue" in table_id file_0_view_0.
- Parameter $d_i$ is mapped from column "Demand" in table_id file_0_view_0.
- Parameter $I_i$ is mapped from column "Initial Inventory" in table_id file_0_view_0.