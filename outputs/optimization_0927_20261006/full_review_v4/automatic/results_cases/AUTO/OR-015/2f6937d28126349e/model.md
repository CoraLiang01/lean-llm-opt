#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of products classified under ‘Aalop’ (from column ‘Product Name’ in table_id: file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Demand for product $i$ during the sales horizon (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
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

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - Index set $I$: All records where ‘Product Name’ starts with ‘Aalop’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’