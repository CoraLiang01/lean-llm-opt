#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified as ‘FAUX’ (from ZARASales.csv, column "Product Name").

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from ZARASales.csv, column "Revenue").
- $d_i$: Demand for product $i \in I$ (from ZARASales.csv, column "Demand").
- $I_i$: Initial inventory for product $i \in I$ (from ZARASales.csv, column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: ZARASales.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where "Product Name" contains or starts with "FAUX"
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $I_i$: Column "Initial Inventory"