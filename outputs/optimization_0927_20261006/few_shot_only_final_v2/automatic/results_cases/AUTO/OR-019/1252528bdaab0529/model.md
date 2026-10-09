**Sets:**  
Let $\mathcal{I}$ be the set of all products in table_id = file_0_view_0 (SalesDataAnalysis.csv) such that the value in column "Product Name" contains '27in'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in file_0_view_0)
- $d_i$: Demand for product $i$ (from column "Demand" in file_0_view_0)
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in file_0_view_0)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill  
  $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{I}$:
1. Demand constraint:  
   $x_i \leq d_i$
2. Inventory constraint:  
   $x_i \leq I_i$
3. Nonnegativity and integrality:  
   $x_i \in \mathbb{Z}_+, \quad x_i \geq 0$

**Data Mapping:**  
- Table: file_0_view_0 (SalesDataAnalysis.csv)
- Product Name: column "Product Name" (filter: contains '27in')
- Revenue: column "Revenue"
- Demand: column "Demand"
- Initial Inventory: column "Initial Inventory"

**Abstract Model Summary:**  
- Sets, parameters, and variables are defined symbolically.
- Objective maximizes total revenue from fulfilled units of all ‘27in’ products.
- Each $x_i$ is bounded above by both demand and initial inventory, and is a nonnegative integer.
- All data columns and table identifiers are mapped exactly as in the source.