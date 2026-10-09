---

**Sets:**  
- $I$ : set of all car models with `Product Name` = 'FDK57' in table `file_0_view_0`.

**Parameters:**  
- $A_i$ : revenue per unit for model $i \in I$, from column `Revenue` in table `file_0_view_0`.
- $d_i$ : demand for model $i \in I$, from column `Demand` in table `file_0_view_0`.
- $I_i$ : initial inventory for model $i \in I$, from column `Initial Inventory` in table `file_0_view_0`.

**Decision Variables:**  
- $x_i$ : number of units of model $i \in I$ to fulfill (integer, $x_i \geq 0$).

**Objective:**  
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**  
- Demand and inventory bounds:
  \[
  0 \leq x_i \leq \min\{d_i, I_i\} \qquad \forall i \in I
  \]
- Integrality:
  \[
  x_i \in \mathbb{Z} \qquad \forall i \in I
  \]

---

**Data Mapping:**  
- Table: `file_0_view_0` (from `BigMartSales.csv`)
- Columns:  
  - `Product Name` (for set $I$ selection: 'FDK57')  
  - `Revenue` (parameter $A_i$)  
  - `Demand` (parameter $d_i$)  
  - `Initial Inventory` (parameter $I_i$)