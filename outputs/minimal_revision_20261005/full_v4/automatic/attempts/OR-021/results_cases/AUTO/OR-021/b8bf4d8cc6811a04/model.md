#### Index Sets

- $I$: Set of all products, indexed by $i$.  
  (Source: table_id = "file_0_view_0", column = "Product Name")

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (Source: table_id = "file_0_view_0", column = "Revenue")
- $d_i$: Total demand for product $i$ over the sales horizon.  
  (Source: table_id = "file_0_view_0", column = "Demand")
- $I_i$: Initial inventory of product $i$.  
  (Source: table_id = "file_0_view_0", column = "Initial Inventory")

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$), for all $i \in I$.

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
  $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

- **table_id:** "file_0_view_0"
- **Product Name** $\rightarrow$ index set $I$
- **Revenue** $\rightarrow$ parameter $A_i$
- **Demand** $\rightarrow$ parameter $d_i$
- **Initial Inventory** $\rightarrow$ parameter $I_i$