**Sets:**  
Let $\mathcal{B}$ be the set of all products where the value in column `Product_Name` begins with "Books".

**Parameters:**  
For each $i \in \mathcal{B}$:
- $A_i$: Revenue per unit of product $i$ (from column `Revenue` in table_id `file_0_view_0`)
- $d_i$: Demand for product $i$ (from column `Demand` in table_id `file_0_view_0`)
- $I_i$: Initial inventory for product $i$ (from column `Initial Inventory` in table_id `file_0_view_0`)

**Decision Variables:**  
For each $i \in \mathcal{B}$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{B}} A_i x_i
$$

**Constraints:**  
For all $i \in \mathcal{B}$:
1. Inventory constraint:  
   $x_i \leq I_i$
2. Demand constraint:  
   $x_i \leq d_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{B}$

**Data Mapping:**  
- Source table: `file_0_view_0` (from `DifferentStoreSales.csv`)
- Product identifier: `Product_Name`
- Revenue parameter: `Revenue`
- Demand parameter: `Demand`
- Initial inventory parameter: `Initial Inventory`