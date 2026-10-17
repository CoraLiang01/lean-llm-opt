#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $a_i$: Revenue per unit for product $i$ (from column "Revenue").
- $d_i$: Demand for product $i$ during the sales cycle (from column "Demand").
- $s_i$: Initial inventory for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ fulfilled (orders fulfilled), $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `MobileSalesDataset.csv`)
- **Index Set $I$:** All rows in column `"Product Name"`
- **Parameter $a_i$:** Column `"Revenue"` in `file_0_view_0`
- **Parameter $d_i$:** Column `"Demand"` in `file_0_view_0`
- **Parameter $s_i$:** Column `"Initial Inventory"` in `file_0_view_0`