#### Sets
- $I$: Set of products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
  (Data: table_id = file_0_view_0, column = Revenue)
- $d_i$: Demand for product $i$.  
  (Data: table_id = file_0_view_0, column = Demand)
- $I_i$: Initial inventory for product $i$.  
  (Data: table_id = file_0_view_0, column = Initial Inventory)

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective
\[
\max \sum_{i \in I} A_i \, x_i
\]

#### Constraints
1. **Demand fulfillment:**  
  $x_i \leq d_i \quad \forall i \in I$

2. **Inventory limit:**  
  $x_i \leq I_i \quad \forall i \in I$

3. **Nonnegativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- $I$: All records in table_id = file_0_view_0
- $A_i$: file_0_view_0, column = Revenue
- $d_i$: file_0_view_0, column = Demand
- $I_i$: file_0_view_0, column = Initial Inventory