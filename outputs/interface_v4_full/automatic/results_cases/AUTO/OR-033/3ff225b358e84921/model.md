#### Abstract Mathematical Optimization Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘Baby’ (from the data).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (parameter).
    - $I_i$: Initial inventory of product $i$ (parameter).
    - $d_i$: Demand for product $i$ (parameter).
    - $x_i$: Number of units of product $i$ to fulfill (decision variable).

##### Variables

$$
x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in \mathcal{I}
$$

##### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

##### Constraints

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

- Table: EuropeSalesRecords.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All rows where ‘Product Name’ has prefix ‘Baby’
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $I_i$: column ‘Initial Inventory’
    - Parameter $d_i$: column ‘Demand’