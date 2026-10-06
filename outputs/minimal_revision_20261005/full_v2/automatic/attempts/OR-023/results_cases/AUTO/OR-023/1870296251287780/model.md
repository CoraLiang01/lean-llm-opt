#### Index Sets

- $I$: Set of all products with Product_Reference starting with ‘ELE-S’ (from table_id: file_0_view_0).

#### Parameters

- $r_i$: Revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: Demand for product $i \in I$ (from column "Demand").
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective

$$
\max \sum_{i \in I} r_i \, x_i
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

3. **Variable Domain**:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **table_id:** file_0_view_0
- **columns used:** 
  - Product_Reference (for set $I$)
  - Revenue (for $r_i$)
  - Demand (for $d_i$)
  - Initial Inventory (for $s_i$)