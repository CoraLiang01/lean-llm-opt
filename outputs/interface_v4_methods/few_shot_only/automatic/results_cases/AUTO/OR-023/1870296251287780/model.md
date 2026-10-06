#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified as ‘ELE-S’ (identified by Product_Reference in the data).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** file_0_view_0 (from SalesStoreoverview.csv)
- **Index Set:** $I$ is defined by all rows where Product_Reference starts with ‘ELE-S’
- **Parameters:**
    - $A_i$ : file_0_view_0, column ‘Revenue’
    - $d_i$ : file_0_view_0, column ‘Demand’
    - $I_i$ : file_0_view_0, column ‘Initial Inventory’
- **Identifiers:** Product_Reference (file_0_view_0, column ‘Product_Reference’) uniquely identifies each $i \in I$.

---

This model maximizes total revenue from all ‘ELE-S’ products, subject to initial inventory and demand constraints, using the exact data columns and identifiers as mapped above.