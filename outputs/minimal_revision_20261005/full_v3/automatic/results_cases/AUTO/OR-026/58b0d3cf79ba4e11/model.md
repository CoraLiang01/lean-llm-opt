#### Sets
- $I$: Index set of all products classified under ‘Fashion’ (from column “Product Name” in table_id: file_0_view_0).

#### Parameters
- $a_i$: Revenue per unit of product $i \in I$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$: Deterministic demand for product $i \in I$ (from column “Demand” in table_id: file_0_view_0).
- $s_i$: Initial inventory for product $i \in I$ (from column “Initial Inventory” in table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective
\[
\max \sum_{i \in I} a_i x_i
\]

#### Constraints
1. **Inventory constraint:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name**: defines set $I$
    - **Revenue**: parameter $a_i$
    - **Demand**: parameter $d_i$
    - **Initial Inventory**: parameter $s_i$