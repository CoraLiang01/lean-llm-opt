#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products with $id\_number$ = 'id999' in table `file_0_view_0`.

**Parameters:**
- $a_i$ : Revenue per unit of product $i$, from column `Revenue` in table `file_0_view_0`.
- $d_i$ : Demand for product $i$ during the sales horizon, from column `Demand` in table `file_0_view_0`.
- $s_i$ : Initial inventory of product $i$, from column `Initial Inventory` in table `file_0_view_0$.

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

**Variable Domains:**
- $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `OnlineRetailSalesDataset.csv`)
- **Index Set:** $I$ = all rows where `id_number` = 'id999'
- **Parameter $a_i$:** column `Revenue`
- **Parameter $d_i$:** column `Demand`
- **Parameter $s_i$:** column `Initial Inventory`