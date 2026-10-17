#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘27in’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint for each product:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- **Index Set $I$:** All rows in table_id file_0_view_0 where ‘Product Name’ has prefix ‘27in’.
- **Parameter $A_i$:** ‘Revenue’ column in table_id file_0_view_0.
- **Parameter $d_i$:** ‘Demand’ column in table_id file_0_view_0.
- **Parameter $I_i$:** ‘Initial Inventory’ column in table_id file_0_view_0.
- **Decision Variable $x_i$:** Number of units fulfilled for each $i \in I$.

All data is sourced from table_id file_0_view_0, columns ‘Product Name’, ‘Revenue’, ‘Demand’, and ‘Initial Inventory’, filtered to rows where ‘Product Name’ has prefix ‘27in’.