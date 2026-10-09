**Abstract Mathematical Model**

**Index Sets:**  
- $S$: Set of all products $i$ such that Product_Reference in table file_0_view_0 starts with 'ELE-S'.

**Parameters:**  
- $A_i$: Revenue per unit of product $i \in S$ (from column 'Revenue')
- $d_i$: Demand for product $i \in S$ (from column 'Demand')
- $I_i$: Initial Inventory for product $i \in S$ (from column 'Initial Inventory')

**Decision Variables:**  
- $x_i \in \mathbb{Z}_+$: Number of units of product $i \in S$ to fulfill

**Objective:**  
\[
\max \sum_{i \in S} A_i \cdot x_i
\]

**Constraints:**  
\[
\forall i \in S:
\]
\[
x_i \leq d_i
\]
\[
x_i \leq I_i
\]
\[
x_i \geq 0
\]
\[
x_i \in \mathbb{Z}
\]

**Data Mapping:**  
- Table: file_0_view_0 (source: SalesStoreoverview.csv)
- Index set $S$: All records where Product_Reference starts with 'ELE-S'
- $A_i$: file_0_view_0, column 'Revenue'
- $d_i$: file_0_view_0, column 'Demand'
- $I_i$: file_0_view_0, column 'Initial Inventory'

**Selection Rule:**  
- The set $S$ is defined by selecting all records in file_0_view_0 where Product_Reference begins with 'ELE-S'. No other filters are applied. All such records are included.