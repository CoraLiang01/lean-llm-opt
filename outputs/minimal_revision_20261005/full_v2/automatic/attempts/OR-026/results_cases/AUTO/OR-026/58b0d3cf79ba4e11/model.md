#### Sets
- $I$: Set of all products classified under ‘Fashion’ (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
  [Data Mapping: table_id = file_0_view_0, column = Revenue]
- $d_i$: Deterministic total demand for product $i$.  
  [Data Mapping: table_id = file_0_view_0, column = Demand]
- $I_i$: Initial inventory for product $i$.  
  [Data Mapping: table_id = file_0_view_0, column = Initial Inventory]

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \, x_i
\]

#### Constraints
1. **Inventory Constraint:**  
  $x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) and the index set $I$ are sourced from:
    - **table_id:** file_0_view_0
    - **columns:** Product Name, Revenue, Demand, Initial Inventory

No additional constraints or synthetic scenario parameters are specified in the query. All variable domains, bounds, and the objective sense are as described above.