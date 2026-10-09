#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$.
- $d_i$: Total demand for product $i \in I$ over the sales horizon.
- $I_i$: Initial inventory for product $i \in I$.

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All records in table_id: `file_0_view_0` (WomenClothingEcommerceSalesData.csv).
- **Parameter $A_i$:** Column `Revenue` in table_id: `file_0_view_0`.
- **Parameter $d_i$:** Column `Demand` in table_id: `file_0_view_0`.
- **Parameter $I_i$:** Column `Initial Inventory` in table_id: `file_0_view_0`.