#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : set of all pizza types (indexed by $i$)

**Parameters:**
- $A_i$ : revenue per unit of pizza type $i$ (from CSVQA_DATA: file_0_view_0, column 'Revenue')
- $d_i$ : total demand for pizza type $i$ (from CSVQA_DATA: file_0_view_0, column 'Demand')
- $I_i$ : initial inventory available for pizza type $i$ (from CSVQA_DATA: file_0_view_0, column 'Initial Inventory')

**Decision Variables:**
- $x_i$ : number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
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

- $I$ : All unique values in 'Product Name' from CSVQA_DATA: file_0_view_0
- $A_i$ : 'Revenue' column, indexed by 'Product Name', from CSVQA_DATA: file_0_view_0
- $d_i$ : 'Demand' column, indexed by 'Product Name', from CSVQA_DATA: file_0_view_0
- $I_i$ : 'Initial Inventory' column, indexed by 'Product Name', from CSVQA_DATA: file_0_view_0

All data is sourced from table_id file_0_view_0, columns 'Product Name', 'Revenue', 'Demand', and 'Initial Inventory'.