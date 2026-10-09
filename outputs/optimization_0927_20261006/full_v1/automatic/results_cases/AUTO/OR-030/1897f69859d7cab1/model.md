#### Abstract Mathematical Optimization Model

Let:

- $\mathcal{I}$: Index set of all car models classified as ‘FDK57’ (from the dataset).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of car model $i$ (from column ‘Revenue’).
    - $d_i$: Total deterministic demand for car model $i$ (from column ‘Demand’).
    - $I_i$: Initial inventory for car model $i$ (from column ‘Initial Inventory’).
    - $x_i$: Decision variable; number of units of car model $i$ to fulfill.

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
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from BigMartSales.csv)
- **Index set:** All rows where ‘Product Name’ has prefix ‘FDK57’
- **Parameters:**
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column
- **Variables:** $x_i$ for each $i \in \mathcal{I}$

No additional constraints or scenario-specific bounds are imposed by the query. All data columns and identifiers are mapped as above.