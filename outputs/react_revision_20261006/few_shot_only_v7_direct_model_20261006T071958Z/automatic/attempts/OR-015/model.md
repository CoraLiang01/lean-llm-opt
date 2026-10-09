##### Abstract Mathematical Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products classified under ‘Aalop’ in the current data.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ over the sales horizon (from column ‘Demand’)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints:**
1. **Inventory Constraint:**  
   $x_i \leq I_i \quad \forall i \in \mathcal{I}$

2. **Demand Constraint:**  
   $x_i \leq d_i \quad \forall i \in \mathcal{I}$

3. **Nonnegativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

##### Data Mapping

- **Index Set $\mathcal{I}$:** All records in table_id = file_0_view_0 where ‘Product Name’ is classified under ‘Aalop’ (as per the user’s definition and current data).
- **Parameter $A_i$:** file_0_view_0, column ‘Revenue’
- **Parameter $d_i$:** file_0_view_0, column ‘Demand’
- **Parameter $I_i$:** file_0_view_0, column ‘Initial Inventory’