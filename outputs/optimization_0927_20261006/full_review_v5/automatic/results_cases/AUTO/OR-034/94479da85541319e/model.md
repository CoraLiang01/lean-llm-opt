#### Abstract Optimization Model

**Index Sets:**
- $I$: Set of all baked goods (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column ‘Revenue’).
- $d_i$: Total demand for baked good $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for baked good $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, for all $i \in I$.

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

- **Source Table:** `Frenchbakerydailysales.csv` (table_id: `file_0_view_0`)
- **Index Set:** $I$ corresponds to all rows in column `Product Name`.
- **Parameters:**
  - $A_i$: column `Revenue`
  - $d_i$: column `Demand`
  - $I_i$: column `Initial Inventory`
- **Variables:** $x_i$ defined for each $i \in I$.

No additional filters or restrictions were applied; all records in the table are included.