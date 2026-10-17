#### Index Sets

- $I$: set of all products, indexed by $i$.

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: total demand for product $i$ over the sales horizon (from column "Demand", table_id: file_0_view_0).
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, integer, $x_i \geq 0$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

   (Equivalently, two constraints per product:)
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. Integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id: file_0_view_0, columns "Revenue", "Demand", and "Initial Inventory", respectively, for all products (no filter; all records used).