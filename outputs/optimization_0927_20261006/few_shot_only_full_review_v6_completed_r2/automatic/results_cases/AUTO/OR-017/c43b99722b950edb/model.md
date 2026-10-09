**Sets:**
- Let $\mathcal{Z}$ be the set of all products with SKU starting with ‘ZZ’ in table file_0_view_0.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in \mathcal{Z}$ (file_0_view_0, column: Revenue)
- $d_i$: Demand for product $i \in \mathcal{Z}$ (file_0_view_0, column: Demand)
- $I_i$: Initial Inventory for product $i \in \mathcal{Z}$ (file_0_view_0, column: Initial Inventory)

**Decision Variables:**
- $x_i$: Number of units of product $i \in \mathcal{Z}$ to fulfill  
  $x_i \in \mathbb{Z}_+, \forall i \in \mathcal{Z}$

**Objective:**
\[
\max \sum_{i \in \mathcal{Z}} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{Z} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{Z} \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{Z}
\end{align*}
\]

**Data Mapping:**  
- Table: file_0_view_0 (RetailStoreSalesTransactions(ScannerData).csv)
- Product selection: All rows where SKU starts with ‘ZZ’
- Revenue: column ‘Revenue’
- Demand: column ‘Demand’
- Initial Inventory: column ‘Initial Inventory’
- Decision variable $x_i$ defined for each such product $i$

**End of Model**