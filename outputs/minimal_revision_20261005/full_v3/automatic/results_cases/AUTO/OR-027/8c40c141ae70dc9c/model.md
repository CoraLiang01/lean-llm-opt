#### Sets
- $I$: Index set of products classified under ‘Organ’.

#### Parameters
- $A_i$: Revenue per unit of product $i \in I$.  
  (Data: table_id = file_0_view_0, column = "Revenue")
- $d_i$: Total deterministic demand for product $i \in I$.  
  (Data: table_id = file_0_view_0, column = "Demand")
- $I_i$: Initial inventory for product $i \in I$.  
  (Data: table_id = file_0_view_0, column = "Initial Inventory")

#### Decision Variables
- $x_i$: Number of units of product $i \in I$ to fulfill.  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint:**  
  $x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Set $I$**: All records in table_id = file_0_view_0, column = "Sub Category" (filtered to contain ‘Organ’)
- **Parameter $A_i$**: table_id = file_0_view_0, column = "Revenue"
- **Parameter $d_i$**: table_id = file_0_view_0, column = "Demand"
- **Parameter $I_i$**: table_id = file_0_view_0, column = "Initial Inventory"