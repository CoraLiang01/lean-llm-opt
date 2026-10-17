#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘Aalop’.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** RestaurantSalesreport.csv
- **Index Set $I$:** All records where ‘Product Name’ has prefix ‘Aalop’ (filter: Product Name prefix ‘Aalop’).
- **Parameter $A_i$:** Column ‘Revenue’ in RestaurantSalesreport.csv.
- **Parameter $d_i$:** Column ‘Demand’ in RestaurantSalesreport.csv.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in RestaurantSalesreport.csv.