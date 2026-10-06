#### Sets
- $I$: Index set of all products classified under ‘Baby’ (from column “Product Name” in table_id: file_0_view_0).

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$: Total deterministic demand for product $i$ (from column “Demand” in table_id: file_0_view_0).
- $I_i$: Initial inventory available for product $i$ (from column “Initial Inventory” in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraint**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraint**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Variable Domain**  
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   \]

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name**: Defines set $I$ (all products classified under ‘Baby’)
    - **Revenue**: Parameter $A_i$
    - **Demand**: Parameter $d_i$
    - **Initial Inventory**: Parameter $I_i$