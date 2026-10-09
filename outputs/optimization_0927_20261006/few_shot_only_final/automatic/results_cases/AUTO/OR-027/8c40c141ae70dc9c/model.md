**Index Sets:**  
Let $\mathcal{I}$ be the set of all products in table_id = "file_0_view_0" where "Sub Category" contains "Organ".

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ ("Revenue" column)
- $I_i$: Initial inventory of product $i$ ("Initial Inventory" column)
- $d_i$: Demand for product $i$ ("Demand" column)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

**Objective:**  
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{I}$:
1. Inventory constraint:  
   $x_i \leq I_i$
2. Demand constraint:  
   $x_i \leq d_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_+$

**Data Mapping:**  
- Table: table_id = "file_0_view_0"
- Index set $\mathcal{I}$: All rows where "Sub Category" contains "Organ"
- $A_i$: "Revenue" column
- $I_i$: "Initial Inventory" column
- $d_i$: "Demand" column