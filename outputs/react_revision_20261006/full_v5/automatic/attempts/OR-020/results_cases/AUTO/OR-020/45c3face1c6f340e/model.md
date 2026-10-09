#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products, indexed by $i$ (from all rows in table_id file_0_view_0, column "Product Name").

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All values in column "Product Name" from table_id file_0_view_0 (SalesDatainBusinesses.csv).
- **Parameter $A_i$:** "Revenue" column, table_id file_0_view_0.
- **Parameter $d_i$:** "Demand" column, table_id file_0_view_0.
- **Parameter $I_i$:** "Initial Inventory" column, table_id file_0_view_0.
- **Variable $x_i$:** Decision variable for each $i \in I$.

All data is mapped directly from the specified columns in table_id file_0_view_0 (SalesDatainBusinesses.csv).