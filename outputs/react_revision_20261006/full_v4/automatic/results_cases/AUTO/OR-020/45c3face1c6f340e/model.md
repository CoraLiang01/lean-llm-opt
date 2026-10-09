#### Mathematical Optimization Model

**Index Set:**
- $I$: set of all products, indexed by $i$ (from all rows in table_id = file_0_view_0, column "Product Name")

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from "Revenue")
- $d_i$: deterministic demand for product $i$ (from "Demand")
- $I_i$: initial inventory available for product $i$ (from "Initial Inventory")

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$, with $x_i \geq 0$ and integer

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory availability:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All unique values in "Product Name" from table_id = file_0_view_0
- **Parameter $A_i$:** "Revenue" column, table_id = file_0_view_0
- **Parameter $d_i$:** "Demand" column, table_id = file_0_view_0
- **Parameter $I_i$:** "Initial Inventory" column, table_id = file_0_view_0
- **Decision Variable $x_i$:** Defined for each $i \in I$ as above

All data is sourced from table_id = file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory".