#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of all pizza types (indexed by $i$).

**Parameters:**
- $A_i$: revenue per unit for pizza type $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: total demand for pizza type $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: initial inventory for pizza type $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: number of units of pizza type $i$ to fulfill.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory and Demand Bounds:**
   \[
   0 \leq x_i \leq \min\{I_i,\, d_i\} \quad \forall i \in I
   \]
   (Equivalently, $x_i \leq I_i$ and $x_i \leq d_i$ for all $i$.)

2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

---

**Data Mapping:**

- All pizza types, with parameters $A_i$, $d_i$, $I_i$, and identifiers $i$, are taken from:
  - table_id: file_0_view_0 (PizzaSalesDataset.csv)
  - columns: ‘Product Name’ (pizza type identifier), ‘Revenue’, ‘Demand’, ‘Initial Inventory’
  - No filters applied; all records used as returned by CSVQA.