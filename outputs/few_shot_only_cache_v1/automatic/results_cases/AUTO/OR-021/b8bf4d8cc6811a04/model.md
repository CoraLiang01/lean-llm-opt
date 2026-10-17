#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `Salesofsummerclothes.csv`)
- **Index Set $I$:** All rows in column `Product Name`
- **Parameter $A_i$:** Column `Revenue` in `file_0_view_0`
- **Parameter $d_i$:** Column `Demand` in `file_0_view_0`
- **Parameter $I_i$:** Column `Initial Inventory` in `file_0_view_0`
- **Decision Variable $x_i$:** Defined for each $i \in I$ (each row/product in the table)

All rows and columns are included; no selection or filtering is applied.