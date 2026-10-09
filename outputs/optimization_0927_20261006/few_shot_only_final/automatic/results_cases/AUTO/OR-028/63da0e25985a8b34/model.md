**Abstract Mathematical Model**

**Index Set:**  
Let $\mathcal{I}$ be the set of products, indexed by $i$, corresponding to all records in  
table_id: `file_0_view_0`, column: `Product Name`.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (table_id: `file_0_view_0`, column: `Revenue`)
- $d_i$: Demand for product $i$ (table_id: `file_0_view_0`, column: `Demand`)
- $I_i$: Initial Inventory for product $i$ (table_id: `file_0_view_0`, column: `Initial Inventory`)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, with $x_i \geq 0$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**
1. **Inventory Constraint:**  
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$
2. **Demand Constraint:**  
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$
3. **Non-negativity:**  
   $$
   x_i \geq 0, \quad \forall i \in \mathcal{I}
   $$

**Data Mapping:**  
- Index set $\mathcal{I}$: All records in table_id: `file_0_view_0`, column: `Product Name`
- Parameter $A_i$: table_id: `file_0_view_0`, column: `Revenue`
- Parameter $d_i$: table_id: `file_0_view_0`, column: `Demand`
- Parameter $I_i$: table_id: `file_0_view_0`, column: `Initial Inventory`