#### Symbolic Mathematical Model

Let:
- $\mathcal{I}$ = set of all car models classified under ‘FDK57’ (indexed by $i$)
- $A_i$ = revenue per unit for car model $i$
- $d_i$ = deterministic demand for car model $i$
- $I_i$ = initial inventory for car model $i$
- $x_i$ = number of units of car model $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
3. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

#### Data Mapping

- Index set $\mathcal{I}$: All records in table_id file_0_view_0 where Product Name has prefix ‘FDK57’ (from column "Product Name" in BigMartSales.csv)
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $I_i$: "Initial Inventory" column in table_id file_0_view_0
- Variable $x_i$: quantity fulfilled for each $i \in \mathcal{I}$

All data is sourced from table_id file_0_view_0, columns: "Product Name", "Revenue", "Demand", "Initial Inventory" in BigMartSales.csv.