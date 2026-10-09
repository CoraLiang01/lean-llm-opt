**Sets:**  
- $I$: Index set of all products where `Product Name` in table `file_0_view_0` begins with "TABLET_".

**Parameters:**  
- $A_i$: Revenue per unit of product $i$, from column `Revenue` in table `file_0_view_0`.
- $d_i$: Demand for product $i$, from column `Demand` in table `file_0_view_0`.
- $I_i$: Initial inventory for product $i$, from column `Initial Inventory` in table `file_0_view_0`.

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**  
1. **Inventory and Demand Fulfillment:**  
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

2. **Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

**Data Mapping:**  
- Table: `file_0_view_0` (from `SmartphoneRetailOutletSalesData.csv`)
- Columns:  
  - `Product Name` (for set $I$ selection, prefix "TABLET_")  
  - `Revenue` (parameter $A_i$)  
  - `Demand` (parameter $d_i$)  
  - `Initial Inventory` (parameter $I_i$)

---

**Complete Model:**  
$$
\begin{align*}
\max_{x_i} \quad & \sum_{i \in I} A_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_+, \quad \forall i \in I
\end{align*}
$$