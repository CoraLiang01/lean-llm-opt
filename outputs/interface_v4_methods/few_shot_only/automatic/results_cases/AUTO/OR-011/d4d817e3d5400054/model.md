#### Abstract Mathematical Model

**Index Set:**

- $I$ : Set of all products classified under ‘id999’ (from `OnlineRetailSalesDataset.csv`, column `id_number`).

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column `Revenue`).
- $d_i$ : Deterministic demand for product $i \in I$ over the sales horizon (from column `Demand`).
- $I_i$ : Initial inventory of product $i \in I$ (from column `Initial Inventory`).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `OnlineRetailSalesDataset.csv`
- **Index Set:** $I$ is defined by all rows where `id_number` = ‘id999’.
- **Parameters:**
  - $A_i$ from column `Revenue`
  - $d_i$ from column `Demand`
  - $I_i$ from column `Initial Inventory`
- **Decision Variables:** $x_i$ for each $i \in I$ (each ‘id999’ product).