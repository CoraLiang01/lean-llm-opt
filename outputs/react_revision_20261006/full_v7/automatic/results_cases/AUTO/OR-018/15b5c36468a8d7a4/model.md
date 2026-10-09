#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘Baby’ in the dataset.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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
3. **Nonnegativity and Integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table ID: file_0_view_0 (from Salesdata.csv)
    - Index set $I$: All rows where ‘Product Name’ has prefix "Baby"
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column