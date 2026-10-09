#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in the current data whose `Product_Reference` begins with "ELE-S".

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column `Revenue`)
- $d_i$: Total demand for product $i$ (from column `Demand`)
- $I_i$: Initial inventory for product $i$ (from column `Initial Inventory`)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. **Inventory constraint:**  
   $x_i \leq I_i \quad \forall i \in I$
2. **Demand constraint:**  
   $x_i \leq d_i \quad \forall i \in I$
3. **Nonnegativity and integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index set $I$:** All records in table_id `file_0_view_0` where `Product_Reference` starts with "ELE-S"
- **Parameter $A_i$:** `Revenue` column in table_id `file_0_view_0`
- **Parameter $d_i$:** `Demand` column in table_id `file_0_view_0`
- **Parameter $I_i$:** `Initial Inventory` column in table_id `file_0_view_0`
- **Decision variable $x_i$:** Number of fulfilled units for each $i \in I$ (to be optimized)