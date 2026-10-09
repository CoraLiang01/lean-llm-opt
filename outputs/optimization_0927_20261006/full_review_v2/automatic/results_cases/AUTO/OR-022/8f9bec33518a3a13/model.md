#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products with ‘27in’ in their name, as identified in column ‘Product Name’ of table_id file_0_view_0.

**Parameters:**
- $A_i$: Revenue per unit of product $i$, from column ‘Revenue’ in table_id file_0_view_0.
- $d_i$: Total demand for product $i$, from column ‘Demand’ in table_id file_0_view_0.
- $I_i$: Initial inventory for product $i$, from column ‘Initial Inventory’ in table_id file_0_view_0.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (from Salesorders.csv)
- Index set $I$: All records where column ‘Product Name’ contains ‘27in’ (FALLBACK_FULL_DATA: all records returned; user must select relevant subset).
- Parameter $A_i$: column ‘Revenue’
- Parameter $d_i$: column ‘Demand’
- Parameter $I_i$: column ‘Initial Inventory’
- Decision variable $x_i$: number of units fulfilled for each $i \in I$