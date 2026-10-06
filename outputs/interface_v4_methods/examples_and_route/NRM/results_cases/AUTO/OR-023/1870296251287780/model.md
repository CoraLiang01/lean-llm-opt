#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products classified under ‘ELE-S’, indexed by $i$.

**Parameters:**
- $r_i$: Revenue per unit of product $i$.  
  Data Mapping: `SalesStoreoverview.csv`, column `Revenue`, key `Product_Reference`.
- $d_i$: Demand for product $i$.  
  Data Mapping: `SalesStoreoverview.csv`, column `Demand`, key `Product_Reference`.
- $s_i$: Initial Inventory of product $i$.  
  Data Mapping: `SalesStoreoverview.csv`, column `Initial Inventory`, key `Product_Reference`.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill.  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   For all $i \in I$,
   \[
   x_i \leq s_i
   \]
2. **Demand Constraint:**  
   For all $i \in I$,
   \[
   x_i \leq d_i
   \]
3. **Non-negativity and Integrality:**  
   For all $i \in I$,
   \[
   x_i \in \mathbb{Z}_{\geq 0}
   \]

---

#### Data Mapping

- $I$ (Product Set): All `Product_Reference` values in `SalesStoreoverview.csv` where `Product_Reference` starts with "ELE-S".
- $r_i$: `Revenue` column, key `Product_Reference`, table_id: `file_0_view_0`
- $d_i$: `Demand` column, key `Product_Reference`, table_id: `file_0_view_0`
- $s_i$: `Initial Inventory` column, key `Product_Reference`, table_id: `file_0_view_0`