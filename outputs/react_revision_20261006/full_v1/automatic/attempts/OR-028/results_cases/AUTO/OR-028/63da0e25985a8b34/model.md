#### Index Sets

- $I$: set of all products, indexed by $i$ (from column "Product Name" in table_id file_0_view_0)

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0)
- $d_i$: total demand for product $i$ (from column "Demand" in table_id file_0_view_0)
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0)

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective

$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints

1. Inventory and Demand Bounds:
   $$
   0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \in I
   $$

#### Data Mapping

- Index set $I$, parameters $A_i$, $d_i$, $I_i$ are mapped from table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", and "Initial Inventory" respectively.