#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all products classified as ‘FAUX’ (from Product Name in table_id file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from Revenue).
- $d_i$: Total demand for product $i \in I$ (from Demand).
- $I_i$: Initial inventory for product $i \in I$ (from Initial Inventory).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (from ZARASales.csv)
- **Index Set:** $I$ = all rows where Product Name has prefix ‘FAUX’
- **Parameters:**
    - $A_i$: Revenue ← column ‘Revenue’
    - $d_i$: Demand ← column ‘Demand’
    - $I_i$: Initial Inventory ← column ‘Initial Inventory’
- **Decision Variables:** $x_i$ for each $i \in I$