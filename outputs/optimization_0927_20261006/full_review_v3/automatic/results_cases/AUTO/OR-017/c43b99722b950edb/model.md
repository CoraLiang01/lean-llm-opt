#### Abstract Mathematical Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘ZZ’ (as identified by SKU).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (parameter, from column ‘Revenue’).
    - $d_i$: Demand for product $i$ (parameter, from column ‘Demand’).
    - $I_i$: Initial inventory of product $i$ (parameter, from column ‘Initial Inventory’).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

**Variables:**
- $x_i \in \mathbb{Z}_+$ for all $i \in \mathcal{I}$

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
\[
x_i \leq I_i \qquad \forall i \in \mathcal{I}
\]
2. Demand constraint:
\[
x_i \leq d_i \qquad \forall i \in \mathcal{I}
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
\]

---

#### Data Mapping

- Table: file_0_view_0 (from RetailStoreSalesTransactions(ScannerData).csv)
- Filter: All rows where SKU has prefix ‘ZZ’ (i.e., products classified under ‘ZZ’)
- Parameters:
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $I_i$: column ‘Initial Inventory’
    - $i \in \mathcal{I}$: identified by column ‘SKU’ with prefix ‘ZZ’