#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘Books’ (from Product_Name in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from Revenue).
- $d_i$: Demand for product $i \in I$ (from Demand).
- $I_i$: Initial inventory for product $i \in I$ (from Initial Inventory).

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

- **Index Set $I$:** All rows in table_id file_0_view_0 where Product_Name has prefix "Books".
- **Parameter $A_i$:** file_0_view_0, column "Revenue".
- **Parameter $d_i$:** file_0_view_0, column "Demand".
- **Parameter $I_i$:** file_0_view_0, column "Initial Inventory".
- **Variable $x_i$:** Decision variable for each $i \in I$.

All data is sourced from DifferentStoreSales.csv, table_id file_0_view_0, with columns as mapped above.