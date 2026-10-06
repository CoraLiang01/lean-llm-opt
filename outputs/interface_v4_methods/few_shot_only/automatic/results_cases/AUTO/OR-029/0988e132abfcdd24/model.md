#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘FAUX’ (from the data source)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = deterministic demand for product $i$ (parameter)
    - $s_i$ = initial inventory for product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. Inventory constraint:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

#### Data Mapping

- Table: file_0_view_0 (from ZARASales.csv)
    - Index set $I$: all rows where the product is classified as ‘FAUX’ (as per user query; if a column for classification exists, filter accordingly)
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $s_i$: column ‘Initial Inventory’