### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products (indexed by $i$), where each product is identified by its "Product Name" from table_id file_0_view_0.

#### Parameters
- $A_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0).
- $d_i$: Total demand for product $i$ over the sales horizon ("Demand", file_0_view_0).
- $I_i$: Initial inventory available for product $i$ ("Initial Inventory", file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Demand Fulfillment Constraint**:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory Constraint**:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$**: All records in table_id file_0_view_0, column "Product Name".
- **Parameter $A_i$**: file_0_view_0, column "Revenue".
- **Parameter $d_i$**: file_0_view_0, column "Demand".
- **Parameter $I_i$**: file_0_view_0, column "Initial Inventory".

All data is retrieved from table_id file_0_view_0, columns "Product Name", "Revenue", "Demand", and "Initial Inventory", with no filters applied (full table used).