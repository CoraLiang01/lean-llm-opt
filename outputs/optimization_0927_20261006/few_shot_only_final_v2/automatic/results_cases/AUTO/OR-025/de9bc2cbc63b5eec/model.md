**Sets:**  
- $I$ : Set of all TABLET products, indexed by $i$.

**Parameters:**  
- $A_i$ : Revenue per unit of product $i$ (from column ‘Revenue’ in table_id = file_0_view_0).
- $d_i$ : Demand for product $i$ (from column ‘Demand’ in table_id = file_0_view_0).
- $I_i$ : Initial inventory for product $i$ (from column ‘Initial Inventory’ in table_id = file_0_view_0).

**Decision Variables:**  
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

**Variable Domains:**  
- $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in I$.

**Objective:**  
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**  
1. **Inventory and Demand Fulfillment:**  
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]

**Data Mapping:**  
- All parameters ($A_i$, $d_i$, $I_i$) and index set $I$ are defined using records from table_id = file_0_view_0, with:
    - $A_i$ from column ‘Revenue’
    - $d_i$ from column ‘Demand’
    - $I_i$ from column ‘Initial Inventory’
    - $I$ is the set of all records where column ‘Product Name’ begins with ‘TABLET_’

---

**Complete Abstract Model:**

Sets:  
- $I$ = set of TABLET products (from table_id = file_0_view_0, ‘Product Name’ starting with ‘TABLET_’)

Parameters:  
- $A_i$ = revenue per unit of product $i$ (file_0_view_0, ‘Revenue’)  
- $d_i$ = demand for product $i$ (file_0_view_0, ‘Demand’)  
- $I_i$ = initial inventory for product $i$ (file_0_view_0, ‘Initial Inventory’)

Variables:  
- $x_i \in \mathbb{Z}_+$, $\forall i \in I$

Objective:  
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:  
\[
0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
\]

**Data Mapping:**  
- All sets and parameters are mapped from table_id = file_0_view_0, using columns ‘Product Name’, ‘Revenue’, ‘Demand’, and ‘Initial Inventory’.  
- $I$ is defined by filtering ‘Product Name’ for entries starting with ‘TABLET_’.