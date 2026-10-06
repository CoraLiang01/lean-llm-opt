#### Sets
- $I$: Set of all product categories (indexed by $i$).  
  (Data Mapping: file_0_view_0, column "Product Name")

#### Parameters
- $A_i$: Revenue per unit for product $i$.  
  (Data Mapping: file_0_view_0, column "Revenue")
- $d_i$: Total deterministic demand for product $i$.  
  (Data Mapping: file_0_view_0, column "Demand")
- $s_i$: Initial inventory available for product $i$.  
  (Data Mapping: file_0_view_0, column "Initial Inventory")

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:**  
  $x_i \leq s_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

- **Set $I$ (Product categories):**  
  file_0_view_0, column "Product Name"
- **Parameter $A_i$ (Revenue):**  
  file_0_view_0, column "Revenue"
- **Parameter $d_i$ (Demand):**  
  file_0_view_0, column "Demand"
- **Parameter $s_i$ (Initial Inventory):**  
  file_0_view_0, column "Initial Inventory"