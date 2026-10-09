---

#### Index Sets

- Let $\mathcal{I}$ be the set of all products with "Product Name" beginning with 'S700_' in table_id `file_0_view_0`.

#### Parameters

- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (`Revenue` column)
    - $d_i$: Total demand for product $i$ (`Demand` column)
    - $I_i$: Initial inventory of product $i$ (`Initial Inventory` column)

#### Decision Variables

- For each $i \in \mathcal{I}$:
    - $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints**  
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$

2. **Demand Constraints**  
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$

3. **Non-negativity and Integrality**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- **Source Table:** `file_0_view_0`
- **Columns Used:**
    - `Product Name` (for index set $\mathcal{I}$, filtered by prefix 'S700_')
    - `Revenue` (parameter $A_i$)
    - `Demand` (parameter $d_i$)
    - `Initial Inventory` (parameter $I_i$)