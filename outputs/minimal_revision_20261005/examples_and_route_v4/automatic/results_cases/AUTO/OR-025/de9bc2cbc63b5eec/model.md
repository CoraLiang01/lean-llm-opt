**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of all products with ‘Product Name’ starting with "TABLET" from `file_0_view_0` (SmartphoneRetailOutletSalesData.csv).

**Parameters:**
- $r_i$: Revenue per unit of product $i$, from column `Revenue` in `file_0_view_0`.
- $d_i$: Demand for product $i$, from column `Demand` in `file_0_view_0`.
- $s_i$: Initial inventory for product $i$, from column `Initial Inventory` in `file_0_view_0$.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\forall i \in I: \quad x_i \leq d_i$
2. **Inventory limit:**  
   $\forall i \in I: \quad x_i \leq s_i$
3. **Nonnegativity and integrality:**  
   $\forall i \in I: \quad x_i \in \mathbb{Z}_{\geq 0}$

---

**Data Mapping**

- $I$: All rows in `file_0_view_0` where `Product Name` starts with "TABLET".
- $r_i$: `file_0_view_0`, column `Revenue`, key `Product Name`.
- $d_i$: `file_0_view_0`, column `Demand`, key `Product Name`.
- $s_i$: `file_0_view_0`, column `Initial Inventory`, key `Product Name`.