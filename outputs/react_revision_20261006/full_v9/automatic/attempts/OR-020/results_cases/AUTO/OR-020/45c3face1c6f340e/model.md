#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory availability:** 
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All product identifiers from table_id: `file_0_view_0`, column: `"Product Name"`.
- **Parameter $A_i$:** Revenue per unit from table_id: `file_0_view_0`, column: `"Revenue"`.
- **Parameter $d_i$:** Demand from table_id: `file_0_view_0`, column: `"Demand"`.
- **Parameter $I_i$:** Initial Inventory from table_id: `file_0_view_0`, column: `"Initial Inventory"`.
- **Decision Variable $x_i$:** Number of units to fulfill for each product $i \in I$.

All parameters are mapped directly from the specified columns in the current data source.