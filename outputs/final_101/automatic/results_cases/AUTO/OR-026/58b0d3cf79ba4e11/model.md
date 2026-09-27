#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified as ‘Fashion’ (from SupermarketSales.csv, Category = ‘Fashion’).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (SupermarketSales.csv, column ‘Revenue’).
- $d_i$: Total deterministic demand for product $i \in I$ (SupermarketSales.csv, column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (SupermarketSales.csv, column ‘Initial Inventory’).

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
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: SupermarketSales.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where ‘Category’ = ‘Fashion’
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $I_i$: column ‘Initial Inventory’