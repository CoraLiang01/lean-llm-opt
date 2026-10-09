**Sets:**  
Let $\mathcal{B}$ be the set of all products classified as ‘Baby’ in  
table_id = file_0_view_0, column = Product Name.

**Parameters:**  
For each $i \in \mathcal{B}$:
- $A_i$: Revenue per unit of product $i$ (from table_id = file_0_view_0, column = Revenue)
- $d_i$: Demand for product $i$ (from table_id = file_0_view_0, column = Demand)
- $I_i$: Initial inventory of product $i$ (from table_id = file_0_view_0, column = Initial Inventory)

**Decision Variables:**  
For each $i \in \mathcal{B}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$$
\max \sum_{i \in \mathcal{B}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{B}$:
1. Inventory constraint:  
   $x_i \leq I_i$
2. Demand constraint:  
   $x_i \leq d_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{B}$

**Data Mapping:**  
- Set $\mathcal{B}$: All records in table_id = file_0_view_0 where column ‘Product Name’ indicates a ‘Baby’ product.
- Parameter $A_i$: table_id = file_0_view_0, column = Revenue
- Parameter $d_i$: table_id = file_0_view_0, column = Demand
- Parameter $I_i$: table_id = file_0_view_0, column = Initial Inventory