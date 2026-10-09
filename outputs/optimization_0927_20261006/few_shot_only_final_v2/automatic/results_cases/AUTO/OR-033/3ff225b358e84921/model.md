**Sets:**  
Let $\mathcal{I}$ be the set of all products classified as "Baby" in  
file_0_view_0["Product Name"].

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$ = revenue per unit of product $i$ (file_0_view_0["Revenue"])
- $d_i$ = demand for product $i$ (file_0_view_0["Demand"])
- $I_i$ = initial inventory of product $i$ (file_0_view_0["Initial Inventory"])

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$ = number of units of product $i$ to fulfill  
  (domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$)

**Objective:**  
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**  
For all $i \in \mathcal{I}$:
1. Demand constraint:  
   \[
   x_i \leq d_i
   \]
2. Inventory constraint:  
   \[
   x_i \leq I_i
   \]
3. Non-negativity and integrality:  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

**Data Mapping:**  
- Table: file_0_view_0  
- Columns:  
  - "Product Name" (for set $\mathcal{I}$)  
  - "Revenue" (for $A_i$)  
  - "Demand" (for $d_i$)  
  - "Initial Inventory" (for $I_i$)