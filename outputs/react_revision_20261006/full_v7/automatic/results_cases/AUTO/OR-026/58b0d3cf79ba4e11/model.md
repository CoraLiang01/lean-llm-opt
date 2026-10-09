#### Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified under ‘Fashion’ (from column "Product Name" in table_id file_0_view_0).

**Parameters:**
- $a_i$: Revenue per unit of product $i \in I$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column "Demand", table_id file_0_view_0).
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. Inventory bounds:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand bounds:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (SupermarketSales.csv, filtered to ‘Fashion’ products)
    - Index set $I$: column "Product Name"
    - Parameter $a_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"