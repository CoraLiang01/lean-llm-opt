**Sets**  
Let $I$ be the set of all products with names beginning with ‘S700_’ as returned from table_id = file_0_view_0, column "Product Name".

**Parameters**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from "Revenue", table_id = file_0_view_0)
- $d_i$: Demand for product $i$ (from "Demand", table_id = file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from "Initial Inventory", table_id = file_0_view_0)

**Decision Variables**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints**
1. **Inventory constraint:**  
  $x_i \leq I_i \quad \forall i \in I$

2. **Demand constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping**

- **table_id:** file_0_view_0  
- **columns:**  
  - "Product Name" $\rightarrow$ index set $I$  
  - "Revenue" $\rightarrow$ parameter $A_i$  
  - "Demand" $\rightarrow$ parameter $d_i$  
  - "Initial Inventory" $\rightarrow$ parameter $I_i$