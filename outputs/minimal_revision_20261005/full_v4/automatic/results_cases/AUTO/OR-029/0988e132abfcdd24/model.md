**Sets**  
Let $I$ be the set of all products in the dataset with "Product Name" beginning with "FAUX".

**Parameters**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

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
  $x_i \in \mathbb{Z}_+ \quad \forall i \in I$

---

**Data Mapping**

- **Table:** file_0_view_0 (from ZARASales.csv)
    - **Product Name:** defines set $I$ (all rows where "Product Name" starts with "FAUX")
    - **Revenue:** parameter $A_i$
    - **Demand:** parameter $d_i$
    - **Initial Inventory:** parameter $I_i$