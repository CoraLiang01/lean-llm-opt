#### Symbolic Mathematical Model

Let:
- $I$ = set of all products with $\text{Sub Category}$ containing "Organ" (from source data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
x_i \leq d_i \quad \forall i \in I
\]
\[
x_i \leq s_i \quad \forall i \in I
\]
\[
x_i \geq 0 \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z} \quad \forall i \in I
\]

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 with "Sub Category" containing "Organ" in SupermartGrocerySales-RetailAnalyticsDataset.csv
- Parameter $A_i$: column "Revenue", table_id file_0_view_0
- Parameter $d_i$: column "Demand", table_id file_0_view_0
- Parameter $s_i$: column "Initial Inventory", table_id file_0_view_0
- Variable $x_i$: defined for each $i \in I$ as above

All parameters and index sets are mapped directly from the specified columns and filter in the source table.