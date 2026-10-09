#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products where the value in column `Product Name` begins with "TABLET_" in table `file_0_view_0`.

**Parameters:**
- $a_i$ : Revenue per unit of product $i \in I$, from column `Revenue` in table `file_0_view_0$.
- $d_i$ : Deterministic demand for product $i \in I$, from column `Demand` in table `file_0_view_0$.
- $s_i$ : Initial inventory for product $i \in I$, from column `Initial Inventory` in table `file_0_view_0$.

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\forall i \in I: \quad x_i \leq d_i$
2. **Inventory limit:**  
   $\forall i \in I: \quad x_i \leq s_i$
3. **Non-negativity and integrality:**  
   $\forall i \in I: \quad x_i \in \mathbb{Z}_+, \ x_i \geq 0$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `SmartphoneRetailOutletSalesData.csv`)
- **Index Set $I$:** All rows where `Product Name` starts with "TABLET_"
- **Parameter $a_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`