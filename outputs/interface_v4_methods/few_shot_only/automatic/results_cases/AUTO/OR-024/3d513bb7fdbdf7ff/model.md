#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products with identifier starting with ‘S700_’ (from column ‘Product Name’).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
    - $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).
    - $d_i$: Demand for product $i$ (from column ‘Demand’).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

**Objective:**
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**
1. Inventory constraint:
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- Table: file_0_view_0 (from SampleSalesData.csv)
    - Index set $\mathcal{I}$: All rows where ‘Product Name’ starts with ‘S700_’
    - Parameter $A_i$: Column ‘Revenue’
    - Parameter $I_i$: Column ‘Initial Inventory’
    - Parameter $d_i$: Column ‘Demand’