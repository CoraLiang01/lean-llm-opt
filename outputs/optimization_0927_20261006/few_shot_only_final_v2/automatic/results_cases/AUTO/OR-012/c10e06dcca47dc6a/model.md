**Sets:**  
- $I$: Set of all products (indexed by $i$)

**Parameters:**  
- $A_i$: Revenue per unit of product $i$  
- $d_i$: Demand for product $i$  
- $I_i$: Initial inventory for product $i$

**Decision Variables:**  
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$  
  (Number of units of product $i$ to fulfill)

**Objective:**  
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**  
1. Inventory and Demand Fulfillment:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

**Data Mapping:**  
- Table: `file_0_view_0`  
- Product identifier: `Product Name`  
- Revenue parameter: `Revenue`  
- Demand parameter: `Demand`  
- Initial inventory parameter: `Initial Inventory`