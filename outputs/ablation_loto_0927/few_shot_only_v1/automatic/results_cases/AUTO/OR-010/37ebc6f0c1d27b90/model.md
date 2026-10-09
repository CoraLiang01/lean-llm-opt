#### Index Sets

- $I$: Set of all products, where each product $i \in I$ corresponds to a unique "Product Name" in table_id `file_0_view_0`.

#### Parameters

- $A_i$: Revenue per unit for product $i$, from column "Revenue" in table_id `file_0_view_0`.
- $d_i$: Total deterministic demand for product $i$ over the sales cycle, from column "Demand" in table_id `file_0_view_0`.
- $I_i$: Initial inventory available for product $i$, from column "Initial Inventory" in table_id `file_0_view_0$.

#### Decision Variables

- $x_i$: Number of orders fulfilled for product $i$.
  - Domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$ (non-negative integers)

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraints** (cannot fulfill more than available inventory):
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraints** (cannot fulfill more than realized demand):
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All unique values in column "Product Name" of table_id `file_0_view_0`.
- **Parameter $A_i$**: Value from column "Revenue" for product $i$ in table_id `file_0_view_0`.
- **Parameter $d_i$**: Value from column "Demand" for product $i$ in table_id `file_0_view_0`.
- **Parameter $I_i$**: Value from column "Initial Inventory" for product $i$ in table_id `file_0_view_0`.