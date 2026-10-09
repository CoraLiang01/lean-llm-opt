#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Total demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory available for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory availability:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SalesDatainBusinesses.csv`)
- **Columns:**
  - Product identifier: `"Product Name"`
  - Revenue per unit: `"Revenue"`
  - Demand: `"Demand"`
  - Initial inventory: `"Initial Inventory"`
- **Selection:** All rows (no filter; full dataset used as returned by CSVQA).