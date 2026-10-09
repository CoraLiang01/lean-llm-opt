**Sets:**  
Let $\mathcal{I}$ be the set of all products classified as ‘Baby’ in  
table_id = "file_0_view_0", column "Product Name".

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$ = Revenue per unit of product $i$ (from column "Revenue")
- $d_i$ = Demand for product $i$ (from column "Demand")
- $I_i$ = Initial Inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$ = Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{I}$:
1. Demand constraint:  
   $x_i \leq d_i$
2. Inventory constraint:  
   $x_i \leq I_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_+, \quad x_i \geq 0$

**Data Mapping:**  
- Index set $\mathcal{I}$: All rows in table_id = "file_0_view_0" where "Product Name" indicates a ‘Baby’ product  
- $A_i$: "Revenue" column, table_id = "file_0_view_0"  
- $d_i$: "Demand" column, table_id = "file_0_view_0"  
- $I_i$: "Initial Inventory" column, table_id = "file_0_view_0"