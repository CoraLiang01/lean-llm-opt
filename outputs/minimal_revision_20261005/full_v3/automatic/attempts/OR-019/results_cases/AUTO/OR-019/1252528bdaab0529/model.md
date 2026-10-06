#### Index Sets

- $I$: Set of all products classified under ‘27in’ (from column "Product Name" in table_id: file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

Maximize total revenue:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Variable Domain** (fulfillment quantities are non-negative integers):
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name**: Index set $I$
    - **Revenue**: Parameter $A_i$
    - **Demand**: Parameter $d_i$
    - **Initial Inventory**: Parameter $I_i$