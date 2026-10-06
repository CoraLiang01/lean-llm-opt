#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0)
- $d_i$: Total demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0)

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**  
1. **Demand fulfillment:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
2. **Inventory availability:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

**Data Mapping:**  
- Table: file_0_view_0 (from SalesDatainBusinesses.csv)
    - Product index: ‘Product Name’
    - Revenue parameter: ‘Revenue’
    - Demand parameter: ‘Demand’
    - Initial inventory parameter: ‘Initial Inventory’