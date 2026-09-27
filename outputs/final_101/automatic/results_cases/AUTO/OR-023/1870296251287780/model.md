#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified as ‘ELE-S’ (from column Product_Reference).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from column Revenue).
    - $d_i$: Demand for product $i$ (from column Demand).
    - $I_i$: Initial inventory of product $i$ (from column Initial Inventory).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

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

- Table: SalesStoreoverview.csv
    - Index set $\mathcal{I}$: All rows where Product_Reference starts with ‘ELE-S’ (column Product_Reference, table_id: file_0_view_0)
    - Parameter $A_i$: Revenue (column Revenue, table_id: file_0_view_0)
    - Parameter $d_i$: Demand (column Demand, table_id: file_0_view_0)
    - Parameter $I_i$: Initial Inventory (column Initial Inventory, table_id: file_0_view_0)