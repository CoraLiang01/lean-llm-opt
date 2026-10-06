**Abstract Mathematical Model**

**Index Set:**
- $I$: Set of car models with ‘Product Name’ prefix ‘FDK57’ (from file_0_view_0, column ‘Product Name’).

**Parameters:**
- $r_i$: Revenue per unit of car model $i$ (file_0_view_0, ‘Revenue’).
- $d_i$: Demand for car model $i$ (file_0_view_0, ‘Demand’).
- $s_i$: Initial Inventory of car model $i$ (file_0_view_0, ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of car model $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

2. **Demand Constraint:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

3. **Nonnegativity and Integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

**Data Mapping**

- $I$: All rows in file_0_view_0 with ‘Product Name’ prefix ‘FDK57’.
- $r_i$: file_0_view_0, column ‘Revenue’, for each $i$.
- $d_i$: file_0_view_0, column ‘Demand’, for each $i$.
- $s_i$: file_0_view_0, column ‘Initial Inventory’, for each $i$.
- $x_i$: Decision variable for each $i$.

**Source Table:**  
file_0_view_0 (from BigMartSales.csv, filtered to ‘Product Name’ prefix ‘FDK57’)  
Columns: ‘Product Name’, ‘Revenue’, ‘Demand’, ‘Initial Inventory’