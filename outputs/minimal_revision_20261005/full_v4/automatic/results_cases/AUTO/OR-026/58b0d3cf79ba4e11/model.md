#### Index Sets

- $I$: Set of all products classified under ‘Fashion’ in table_id = file_0_view_0, column "Product Name".

#### Parameters

- $r_i$: Revenue per unit of product $i \in I$ (from table_id = file_0_view_0, column "Revenue").
- $d_i$: Total deterministic demand for product $i \in I$ (from table_id = file_0_view_0, column "Demand").
- $s_i$: Initial inventory for product $i \in I$ (from table_id = file_0_view_0, column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective

$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq s_i \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$**: All records in table_id = file_0_view_0, column "Product Name", filtered for products classified under ‘Fashion’.
- **Parameter $r_i$**: table_id = file_0_view_0, column "Revenue".
- **Parameter $d_i$**: table_id = file_0_view_0, column "Demand".
- **Parameter $s_i$**: table_id = file_0_view_0, column "Initial Inventory".