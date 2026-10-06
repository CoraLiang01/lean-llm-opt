#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products classified under ‘ELE-S’, indexed by $i$.

**Parameters:**
- $A_i$ : Revenue per unit of product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Revenue, index: Product_Reference]
- $I_i$ : Initial inventory of product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Initial Inventory, index: Product_Reference]
- $d_i$ : Demand for product $i$.  
  [CSVQA_DATA: file_0_view_0, column: Demand, index: Product_Reference]

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
  $x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- $I$ (product index set): All rows in [file_0_view_0] where Product_Reference starts with ‘ELE-S’.
- $A_i$ (revenue): [file_0_view_0], column ‘Revenue’, indexed by ‘Product_Reference’.
- $I_i$ (initial inventory): [file_0_view_0], column ‘Initial Inventory’, indexed by ‘Product_Reference’.
- $d_i$ (demand): [file_0_view_0], column ‘Demand’, indexed by ‘Product_Reference’.