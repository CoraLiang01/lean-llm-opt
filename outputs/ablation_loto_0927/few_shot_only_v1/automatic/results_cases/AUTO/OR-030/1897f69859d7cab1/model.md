#### Abstract Mathematical Model

**Index Sets**

- $I$ : Set of all car models classified under ‘FDK57’ in `file_0_view_0` (i.e., all rows where `Product Name` = 'FDK57').

**Parameters**

- $A_i$ : Revenue per unit for car model $i \in I$, from column `Revenue` in `file_0_view_0`.
- $d_i$ : Total deterministic demand for car model $i \in I$, from column `Demand` in `file_0_view_0`.
- $I_i$ : Initial inventory for car model $i \in I$, from column `Initial Inventory` in `file_0_view_0$.

**Decision Variables**

- $x_i$ : Number of units of car model $i \in I$ to fulfill (integer, $x_i \geq 0$).

**Objective**

$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints**

1. **Inventory Constraint**  
   $x_i \leq I_i \qquad \forall i \in I$

2. **Demand Constraint**  
   $x_i \leq d_i \qquad \forall i \in I$

3. **Non-negativity and Integrality**  
   $x_i \in \mathbb{Z}_+, \qquad \forall i \in I$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `BigMartSales.csv`)
- **Columns Used:**
  - `Product Name` (for selection of FDK57 models, index set $I$)
  - `Revenue` (parameter $A_i$)
  - `Demand` (parameter $d_i$)
  - `Initial Inventory` (parameter $I_i$)