#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘FAUX’ (see Data Mapping for selection).
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter).
    - $d_i$ = deterministic demand for product $i$ (parameter).
    - $s_i$ = initial inventory for product $i$ (parameter).
    - $x_i$ = number of units of product $i$ to fulfill (decision variable).

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
$$
\max \sum_{i \in I} A_i \cdot x_i
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

- **Table:** file_0_view_0 (from ZARASales.csv)
- **Columns:**
    - Product Name (used to select all products classified as ‘FAUX’)
    - Revenue (parameter $A_i$)
    - Initial Inventory (parameter $s_i$)
    - Demand (parameter $d_i$)
- **Selection:** All rows in file_0_view_0 (FALLBACK_FULL_DATA; see validation note: filter for ‘FAUX’ product family not directly supported by query evidence, so all rows are returned. The index set $I$ should be constructed by selecting all products whose Product Name indicates ‘FAUX’ classification, e.g., names starting with or containing ‘FAUX’.)

No additional constraints or data transformations are imposed beyond those specified above.