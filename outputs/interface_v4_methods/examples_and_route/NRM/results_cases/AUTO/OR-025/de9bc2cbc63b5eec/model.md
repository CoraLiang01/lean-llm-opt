#### Abstract Mathematical Model

Let:
- $I$ = set of all products with 'Product Name' starting with 'TABLET' (see Data Mapping).
- For each $i \in I$:
    - $r_i$ = Revenue for product $i$.
    - $d_i$ = Demand for product $i$.
    - $s_i$ = Initial Inventory for product $i$.
    - $x_i$ = number of units of product $i$ to fulfill (decision variable).

**Parameters and Data Mapping:**
- $I$: All records in SmartphoneRetailOutletSalesData.csv where 'Product Name' starts with 'TABLET' (see table_id: file_0_view_0).
- $r_i$: 'Revenue' column, table_id: file_0_view_0, key: 'Product Name'.
- $d_i$: 'Demand' column, table_id: file_0_view_0, key: 'Product Name'.
- $s_i$: 'Initial Inventory' column, table_id: file_0_view_0, key: 'Product Name'.

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (Product set): All rows in SmartphoneRetailOutletSalesData.csv with 'Product Name' starting with 'TABLET', table_id: file_0_view_0, preserving source_row and 'Product Name'.
- $r_i$: 'Revenue' column, table_id: file_0_view_0, key: 'Product Name'.
- $d_i$: 'Demand' column, table_id: file_0_view_0, key: 'Product Name'.
- $s_i$: 'Initial Inventory' column, table_id: file_0_view_0, key: 'Product Name'.

All indices, parameters, and constraints are mapped directly to the original data columns and rows as returned. No data is omitted or synthesized.