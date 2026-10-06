#### Index Sets

- $I$: Set of all products, where each product $i \in I$ is identified by its "Product Name" from table_id file_0_view_0.

#### Parameters

- $A_i$: Revenue per unit for product $i$, from column "Revenue" in file_0_view_0.
- $d_i$: Total deterministic demand for product $i$ over the sales cycle, from column "Demand" in file_0_view_0.
- $I_i$: Initial inventory for product $i$, from column "Initial Inventory" in file_0_view_0.

#### Decision Variables

- $x_i$: Number of customer orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraints** (cannot fulfill more than available inventory):
   \[
   x_i \leq I_i \quad \forall i \in I
   \]

2. **Demand Constraints** (cannot fulfill more than realized demand):
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

3. **Non-negativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All "Product Name" entries in table_id file_0_view_0.
- **Parameter $A_i$**: "Revenue" column, table_id file_0_view_0.
- **Parameter $d_i$**: "Demand" column, table_id file_0_view_0.
- **Parameter $I_i$**: "Initial Inventory" column, table_id file_0_view_0.