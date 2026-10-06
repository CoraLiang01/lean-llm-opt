**Sets**  
Let $\mathcal{B}$ be the set of all products with $Product\_Name$ classified under ‘Books’ in table_id = file_0_view_0.

**Parameters**  
For each $i \in \mathcal{B}$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables**  
For each $i \in \mathcal{B}$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective**  
\[
\max \sum_{i \in \mathcal{B}} A_i \cdot x_i
\]

**Constraints**  
For all $i \in \mathcal{B}$:
\[
\begin{align*}
x_i &\leq d_i \\
x_i &\leq I_i \\
x_i &\geq 0 \\
x_i &\in \mathbb{Z}
\end{align*}
\]

---

**Data Mapping**

- **table_id:** file_0_view_0
- **Product Name:** Product_Name
- **Revenue:** Revenue
- **Demand:** Demand
- **Initial Inventory:** Initial Inventory

All parameters and index sets are defined directly from the specified columns and table.