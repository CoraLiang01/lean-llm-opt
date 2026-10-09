---

#### Index Sets

- Let $\mathcal{I}$ be the set of all products with `Product Name` beginning with 'S700_' in table `file_0_view_0`.

#### Parameters

- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (`Revenue` column).
- $d_i$: Total demand for product $i \in \mathcal{I}$ (`Demand` column).
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (`Initial Inventory` column).

#### Decision Variables

- $x_i$: Number of units of product $i \in \mathcal{I}$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints:**
   $$
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   $$
3. **Nonnegativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0`
- **Columns:** 
  - `Product Name` (for index set $\mathcal{I}$, filtered by prefix 'S700_')
  - `Revenue` (parameter $A_i$)
  - `Demand` (parameter $d_i$)
  - `Initial Inventory` (parameter $I_i$)