#### Abstract Mathematical Model

Let:

- $I$ = index set of all products with classification ‘ELE-S’ (as identified by all rows in table_id file_0_view_0 where Product_Reference starts with ‘ELE-S’).
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (parameter from column ‘Revenue’)
    - $d_i$ = deterministic demand for product $i$ (parameter from column ‘Demand’)
    - $s_i$ = initial inventory for product $i$ (parameter from column ‘Initial Inventory’)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in I$

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
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- **Index set $I$**: All rows in table_id file_0_view_0 (SalesStoreoverview.csv) where Product_Reference starts with ‘ELE-S’ (FALLBACK_FULL_DATA: all rows returned; selection is explicit).
- **Parameter $r_i$**: file_0_view_0, column ‘Revenue’
- **Parameter $d_i$**: file_0_view_0, column ‘Demand’
- **Parameter $s_i$**: file_0_view_0, column ‘Initial Inventory’
- **Variable $x_i$**: Decision variable for each $i \in I$

No additional constraints or objective terms are imposed by the query. All data and boundaries are mapped directly from the returned table and columns.