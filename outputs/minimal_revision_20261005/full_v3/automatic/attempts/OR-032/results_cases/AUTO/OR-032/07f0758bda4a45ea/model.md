#### Sets
- $I$: Index set of all products classified under ‘Books’ (from column Product_Name in table_id file_0_view_0).

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column Revenue in table_id file_0_view_0).
- $d_i$: Total deterministic demand for product $i$ (from column Demand in table_id file_0_view_0).
- $I_i$: Initial inventory available for product $i$ (from column Initial Inventory in table_id file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraint** (cannot fulfill more than demand):
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality**:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Set $I$**: All rows in table_id file_0_view_0 where Product_Name has prefix "Books".
- **Parameter $A_i$**: Revenue from column Revenue in table_id file_0_view_0.
- **Parameter $d_i$**: Demand from column Demand in table_id file_0_view_0.
- **Parameter $I_i$**: Initial Inventory from column Initial Inventory in table_id file_0_view_0.