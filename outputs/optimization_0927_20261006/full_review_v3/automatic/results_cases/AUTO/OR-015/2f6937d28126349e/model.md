#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘Aalop’ (from column ‘Product Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’ in table_id file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’ in table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

2. Integer Nonnegativity:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Source Table: file_0_view_0 (RestaurantSalesreport.csv)
- Index Set $I$: All records where ‘Product Name’ starts with ‘Aalop’
- Parameter $A_i$: column ‘Revenue’
- Parameter $d_i$: column ‘Demand’
- Parameter $I_i$: column ‘Initial Inventory’
- Decision Variable $x_i$: Number of units fulfilled for each $i \in I$