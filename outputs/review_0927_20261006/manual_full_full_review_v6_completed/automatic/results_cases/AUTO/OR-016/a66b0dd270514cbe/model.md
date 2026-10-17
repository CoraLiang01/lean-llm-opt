#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$), where each product $i$ belongs to a category (e.g., electronics, apparel, homeware, etc.).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraints:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `RetailSalesDataset.csv`)
- **Columns Used:**
  - Product Name: identifies each $i \in I$
  - Revenue: parameter $A_i$
  - Demand: parameter $d_i$
  - Initial Inventory: parameter $I_i$
- **Selection:** All records (no filter applied; full dataset used as returned by CSVQA)