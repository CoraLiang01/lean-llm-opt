**Sets:**  
$\mathcal{I}$: Set of all products in file_0_view_0 with ‘Organ’ in the product classification (see Data Mapping).

**Parameters:**  
$A_i$: Revenue per unit of product $i$, from column ‘Revenue’ in file_0_view_0.  
$d_i$: Demand for product $i$, from column ‘Demand’ in file_0_view_0.  
$I_i$: Initial inventory for product $i$, from column ‘Initial Inventory’ in file_0_view_0.

**Decision Variables:**  
$x_i$: Number of units of product $i$ to fulfill, for all $i \in \mathcal{I}$, with $x_i \geq 0$.

**Objective:**  
$\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i$

**Constraints:**  
1. Inventory and Demand Limits:  
$\quad x_i \leq I_i, \quad \forall i \in \mathcal{I}$  
$\quad x_i \leq d_i, \quad \forall i \in \mathcal{I}$  
2. Non-negativity:  
$\quad x_i \geq 0, \quad \forall i \in \mathcal{I}$

---

**Data Mapping:**  
- Table: file_0_view_0  
- Index set $\mathcal{I}$: All records in file_0_view_0 where the product classification (column name as per user description, e.g., ‘Sub Category’ or similar) contains ‘Organ’  
- $A_i$: file_0_view_0, column ‘Revenue’  
- $d_i$: file_0_view_0, column ‘Demand’  
- $I_i$: file_0_view_0, column ‘Initial Inventory’