#### Abstract Mathematical Model

**Index Set:**
- $I$: set of all pizza types, indexed by $i$

**Parameters:**
- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue")
- $d_i$: total demand for pizza type $i$ (from column "Demand")
- $I_i$: initial inventory for pizza type $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$  
  (number of units of pizza type $i$ to fulfill; non-negative integer)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment Bounds:**  
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Each fulfilled quantity cannot exceed available inventory or demand.)

2. **Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All unique values in column `"Product Name"` of table_id `file_0_view_0`
- **Parameter $A_i$:** Value from column `"Revenue"` for pizza $i$ in table_id `file_0_view_0`
- **Parameter $d_i$:** Value from column `"Demand"` for pizza $i$ in table_id `file_0_view_0`
- **Parameter $I_i$:** Value from column `"Initial Inventory"` for pizza $i$ in table_id `file_0_view_0`
- **Decision Variable $x_i$:** Defined for each $i \in I$ as above

All data is sourced from table_id `file_0_view_0` with columns as specified.