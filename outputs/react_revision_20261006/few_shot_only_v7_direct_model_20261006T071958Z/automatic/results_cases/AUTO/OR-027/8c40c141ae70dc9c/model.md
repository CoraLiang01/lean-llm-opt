#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in the dataset whose ‘Sub Category’ contains the substring “Organ”.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Total demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\displaystyle \max \sum_{i \in I} A_i x_i$

**Constraints:**
1. **Inventory Constraint:**  
   $\displaystyle x_i \leq I_i \quad \forall i \in I$
2. **Demand Constraint:**  
   $\displaystyle x_i \leq d_i \quad \forall i \in I$
3. **Nonnegativity and Integrality:**  
   $\displaystyle x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index Set $I$:** All records in table_id `file_0_view_0` where column `Sub Category` contains the substring “Organ”.
- **Parameter $A_i$:** From column `Revenue` in table_id `file_0_view_0`.
- **Parameter $d_i$:** From column `Demand` in table_id `file_0_view_0`.
- **Parameter $I_i$:** From column `Initial Inventory` in table_id `file_0_view_0`.