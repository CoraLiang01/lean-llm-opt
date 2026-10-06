#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all ‘TABLET’ smartphone models (from `file_0_view_0`, column `Product Name`).

**Parameters:**

- $A_i$ : Revenue per unit for model $i \in I$ (from `file_0_view_0`, column `Revenue`).
- $d_i$ : Deterministic demand for model $i \in I$ (from `file_0_view_0`, column `Demand`).
- $I_i$ : Initial inventory for model $i \in I$ (from `file_0_view_0`, column `Initial Inventory`).

**Decision Variables:**

- $x_i$ : Number of units of model $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq I_i \qquad \forall i \in I
   $$
2. **Demand Constraint:**
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
3. **Nonnegativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SmartphoneRetailOutletSalesData.csv`)
- **Index Set:** $I$ is all rows where the product is classified as ‘TABLET’ (identified by `Product Name` containing the prefix `TABLET_`).
- **Parameters:**
    - $A_i$ : `Revenue`
    - $d_i$ : `Demand`
    - $I_i$ : `Initial Inventory`
- **Decision Variables:** $x_i$ corresponds to each $i \in I$.

No literal data values or record counts are included per instructions.