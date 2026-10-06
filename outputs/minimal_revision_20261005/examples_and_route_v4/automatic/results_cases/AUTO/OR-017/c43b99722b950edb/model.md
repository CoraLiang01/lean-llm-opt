**Mathematical Optimization Model**

**Index Sets:**
- $I$: Set of products with SKU prefix ‘ZZ’ (from all rows in table_id = file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in file_0_view_0, indexed by SKU).
- $d_i$: Demand for product $i$ (from column ‘Demand’ in file_0_view_0, indexed by SKU).
- $s_i$: Initial Inventory for product $i$ (from column ‘Initial Inventory’ in file_0_view_0, indexed by SKU).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

2. **Inventory availability:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

3. **Nonnegativity and integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

**Data Mapping**

- $I$: All rows in table_id = file_0_view_0 (SKU with prefix ‘ZZ’)
- $r_i$: file_0_view_0, column ‘Revenue’, indexed by SKU
- $d_i$: file_0_view_0, column ‘Demand’, indexed by SKU
- $s_i$: file_0_view_0, column ‘Initial Inventory’, indexed by SKU

**Variables:**
- $x_i$: Number of units to fulfill for SKU $i$ (SKU from file_0_view_0)

**Objective:**
- Maximize total revenue from fulfilled units of all ‘ZZ’ products

**Constraints:**
- Cannot fulfill more than demand or available inventory for each product
- All $x_i$ are nonnegative integers