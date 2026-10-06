#### Index Sets

- $I$: Set of all “4U” products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  [From: table_id = file_0_view_0, column = Revenue]

- $d_i$: Total demand for product $i$ over the sales horizon.  
  [From: table_id = file_0_view_0, column = Demand]

- $I_i$: Initial inventory of product $i$.  
  [From: table_id = file_0_view_0, column = Initial Inventory]

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints:**  
  $x_i \leq I_i, \quad \forall i \in I$

2. **Demand Constraints:**  
  $x_i \leq d_i, \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- $I$: All records in table_id = file_0_view_0, column = Product Name, filtered by prefix “4U”
- $A_i$: table_id = file_0_view_0, column = Revenue
- $d_i$: table_id = file_0_view_0, column = Demand
- $I_i$: table_id = file_0_view_0, column = Initial Inventory