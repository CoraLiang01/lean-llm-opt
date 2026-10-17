#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘Books’ (from column ‘Product_Name’ in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, integer and non-negative.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint for each product:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (from DifferentStoreSales.csv)
- **Index Set:** $I$ = all rows where ‘Product_Name’ starts with ‘Books’
- **Parameters:**
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column
- **Decision Variables:** $x_i$ for each $i \in I$