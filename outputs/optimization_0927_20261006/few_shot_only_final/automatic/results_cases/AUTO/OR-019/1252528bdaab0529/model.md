**Sets:**  
Let $\mathcal{I}$ be the set of all products in table_id = file_0_view_0 (SalesDataAnalysis.csv) such that the value in column "Product Name" contains '27in'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
1. Inventory and demand bounds for each product:
   $$
   0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \in \mathcal{I}
   $$

**Data Mapping:**  
- Table: file_0_view_0 (SalesDataAnalysis.csv)
- Product Name: "Product Name"
- Revenue: "Revenue"
- Demand: "Demand"
- Initial Inventory: "Initial Inventory"

**End of Model**