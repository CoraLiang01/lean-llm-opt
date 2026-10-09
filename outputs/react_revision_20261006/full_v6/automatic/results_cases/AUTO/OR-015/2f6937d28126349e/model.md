#### Symbolic Mathematical Model

**Index Set:**
- $I$: set of all products classified under ‘Aalop’ (from the data, all products where Product Name has prefix "Aalop").

**Parameters:**
- $A_i$: revenue per unit of product $i \in I$ (from column "Revenue").
- $d_i$: deterministic demand for product $i \in I$ over the sales horizon (from column "Demand").
- $I_i$: initial inventory of product $i \in I$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
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

**Data Mapping:**

- Table: file_0_view_0 (RestaurantSalesreport.csv)
    - Index set $I$: All rows where "Product Name" has prefix "Aalop"
    - $A_i$: "Revenue"
    - $d_i$: "Demand"
    - $I_i$: "Initial Inventory"