#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all car models classified under ‘FDK57’. (From `file_0_view_0`, filter: `Product Name` with `Classification` = 'FDK57')

**Parameters:**

- $A_i$ : Revenue per unit for car model $i \in I$. (From column `Revenue`)
- $I_i$ : Initial inventory for car model $i \in I$. (From column `Initial Inventory`)
- $d_i$ : Deterministic demand for car model $i \in I$. (From column `Demand`)

**Decision Variables:**

- $x_i$ : Number of units of car model $i \in I$ to fulfill (integer, $0 \leq x_i \leq \min\{I_i, d_i\}$)

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `BigMartSales.csv`)
- **Index Set:** $I$ is defined by all rows where `Classification` = 'FDK57', using `Product Name` as the unique identifier.
- **Parameters:**
  - $A_i$ : `Revenue` column, indexed by `Product Name`
  - $I_i$ : `Initial Inventory` column, indexed by `Product Name`
  - $d_i$ : `Demand` column, indexed by `Product Name`
- **Decision Variables:** $x_i$ corresponds to each $i \in I$.

**Note:** All data elements are referenced by their exact column names and table_id as validated by the CSVQA_DATA bindings. No literal values or record counts are included. All business keys and source structure are preserved.