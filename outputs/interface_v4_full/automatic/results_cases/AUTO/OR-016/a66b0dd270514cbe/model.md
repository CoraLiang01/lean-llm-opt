#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all merchandise products/categories (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $0 \leq x_i \leq \min\{d_i, I_i\}$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory constraint:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: `RetailSalesDataset.csv` (table_id: file_0_view_0)
    - Product/category identifier: `Product Name`
    - Revenue per unit: `Revenue`
    - Demand: `Demand`
    - Initial inventory: `Initial Inventory`