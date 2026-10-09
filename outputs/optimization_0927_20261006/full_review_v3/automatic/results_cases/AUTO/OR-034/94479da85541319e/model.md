#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all baked goods, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Demand for baked good $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for baked good $i$ (from column "Initial Inventory", table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- All parameters ($A_i$, $d_i$, $I_i$) and index set $I$ are defined using all records and the columns "Product Name", "Revenue", "Demand", and "Initial Inventory" from table_id: file_0_view_0 (file: Frenchbakerydailysales.csv), with no additional filters.