**Sets:**  
- $I$: Index set of all products classified under ‘Aalop’ (from column “Product Name”).

**Parameters:**  
- $A_i$: Revenue per unit of product $i$ (from column “Revenue”, table_id: file_0_view_0).
- $d_i$: Demand for product $i$ over the sales horizon (from column “Demand”, table_id: file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column “Initial Inventory”, table_id: file_0_view_0).

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, for all $i \in I$.

**Objective:**  
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**  
1. **Inventory and Demand Limits:**  
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

**Data Mapping:**  
- Table: file_0_view_0 (RestaurantSalesreport.csv)
    - Product Name $\rightarrow$ Set $I$
    - Revenue $\rightarrow$ Parameter $A_i$
    - Demand $\rightarrow$ Parameter $d_i$
    - Initial Inventory $\rightarrow$ Parameter $I_i$