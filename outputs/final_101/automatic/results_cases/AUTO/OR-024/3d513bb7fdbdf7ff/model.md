#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘S700_’ (from column ‘Product Name’).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
    - $d_i$: Total demand for product $i$ (from column ‘Demand’).
    - $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

**Variables:**
- $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in \mathcal{I}$

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

- Table: SampleSalesData.csv
- Table ID: file_0_view_0
- Index set $\mathcal{I}$: All rows where ‘Product Name’ has prefix ‘S700_’ (column ‘Product Name’)
- Parameter $A_i$: column ‘Revenue’
- Parameter $d_i$: column ‘Demand’
- Parameter $I_i$: column ‘Initial Inventory’