##### Mathematical Model

**Index Set:**  
Let $I$ be the set of all products classified under ‘Baby’ in the dataset.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory", table_id: file_0_view_0)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
$\max \sum_{i \in I} A_i x_i$

**Constraints:**  
For all $i \in I$:
- $x_i \leq d_i$  (Demand constraint)
- $x_i \leq I_i$  (Inventory constraint)
- $x_i \geq 0$ and integer

##### Data Mapping

- Index set $I$: All records in table_id: file_0_view_0, column "Product Name" (filtered for ‘Baby’ products)
- $A_i$: table_id: file_0_view_0, column "Revenue"
- $d_i$: table_id: file_0_view_0, column "Demand"
- $I_i$: table_id: file_0_view_0, column "Initial Inventory"