#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘Aalop’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$: Initial Inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Product Name’ starts with ‘Aalop’
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column