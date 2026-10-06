#### Sets
- $I$: Set of all products (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
  (Data: table_id = file_0_view_0, column = "Revenue")
- $d_i$: Total deterministic demand for product $i$ over the sales horizon.  
  (Data: table_id = file_0_view_0, column = "Demand")
- $I_i$: Initial inventory of product $i$.  
  (Data: table_id = file_0_view_0, column = "Initial Inventory")

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  Domain: $x_i \in \mathbb{Z}_+, \forall i \in I$

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:**  
  $x_i \leq I_i, \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i, \quad \forall i \in I$

3. **Nonnegativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Set $I$:** All product names from table_id = file_0_view_0, column = "Product Name"
- **Parameter $A_i$:** table_id = file_0_view_0, column = "Revenue"
- **Parameter $d_i$:** table_id = file_0_view_0, column = "Demand"
- **Parameter $I_i$:** table_id = file_0_view_0, column = "Initial Inventory"