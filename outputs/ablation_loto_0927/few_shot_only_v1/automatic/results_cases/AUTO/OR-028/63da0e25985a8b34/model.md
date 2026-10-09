#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in table_id `file_0_view_0`, column `Product Name`.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (`Revenue`)
- $d_i$: Demand for product $i$ (`Demand`)
- $I_i$: Initial inventory of product $i$ (`Initial Inventory`)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), $x_i \leq \min\{d_i, I_i\}$

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Demand fulfillment:**  
  $\forall i \in I: \quad x_i \leq d_i$

2. **Inventory limit:**  
  $\forall i \in I: \quad x_i \leq I_i$

3. **Non-negativity and integrality:**  
  $\forall i \in I: \quad x_i \in \mathbb{Z}_+, \ x_i \geq 0$

---

#### Data Mapping

- **Index set $I$:**  
  `file_0_view_0`, column `Product Name`
- **Parameter $A_i$:**  
  `file_0_view_0`, column `Revenue`
- **Parameter $d_i$:**  
  `file_0_view_0`, column `Demand`
- **Parameter $I_i$:**  
  `file_0_view_0`, column `Initial Inventory`