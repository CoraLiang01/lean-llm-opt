#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all baked goods (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column "Revenue").
- $d_i$: Total demand for baked good $i$ (from column "Demand").
- $I_i$: Initial inventory for baked good $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: `Frenchbakerydailysales.csv` (table_id: file_0_view_0)
    - $I$: All rows in column "Product Name"
    - $A_i$: Column "Revenue"
    - $d_i$: Column "Demand"
    - $I_i$: Column "Initial Inventory"

No additional filters were applied; all records in the table are included.