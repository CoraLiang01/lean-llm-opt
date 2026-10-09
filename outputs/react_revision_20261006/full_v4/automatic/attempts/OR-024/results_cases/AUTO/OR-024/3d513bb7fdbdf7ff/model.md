#### Mathematical Optimization Model

**Index Set:**  
Let $I$ be the set of all products with names beginning with "S700_" (from column "Product Name" in table_id file_0_view_0).

**Parameters:**  
For each $i \in I$:
- $A_i$: revenue per unit of product $i$ (from column "Revenue", table_id file_0_view_0)
- $d_i$: total demand for product $i$ (from column "Demand", table_id file_0_view_0)
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory", table_id file_0_view_0)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory constraint:**  
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All records in table_id file_0_view_0 where "Product Name" has prefix "S700_"
- **Parameter $A_i$:** "Revenue" column, table_id file_0_view_0
- **Parameter $d_i$:** "Demand" column, table_id file_0_view_0
- **Parameter $I_i$:** "Initial Inventory" column, table_id file_0_view_0
- **Decision Variable $x_i$:** Number of units fulfilled for each $i \in I$