**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products classified under ‘Aalop’ (from RestaurantSalesreport.csv, filtered by Product Name prefix "Aalop").

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Demand for product $i$ during the sales horizon (from column ‘Demand’).
- $s_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in RestaurantSalesreport.csv where Product Name starts with "Aalop" (see file_0_view_0).
- $r_i$: RestaurantSalesreport.csv, column ‘Revenue’, for each $i \in I$.
- $d_i$: RestaurantSalesreport.csv, column ‘Demand’, for each $i \in I$.
- $s_i$: RestaurantSalesreport.csv, column ‘Initial Inventory’, for each $i \in I$.

**Source Table:**  
- table_id: file_0_view_0  
- columns: Product Name, Revenue, Demand, Initial Inventory

**Notes:**  
- No restocking or in-transit inventory is allowed.
- Demand is deterministic and known.
- Only products with Product Name starting with "Aalop" are included.