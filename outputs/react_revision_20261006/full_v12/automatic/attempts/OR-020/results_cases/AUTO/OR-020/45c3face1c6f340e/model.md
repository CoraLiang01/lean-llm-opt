#### Mathematical Model

Let $I$ be the set of all products in the supermarket, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial Inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
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

#### Data Mapping

- Table: file_0_view_0 (from SalesDatainBusinesses.csv)
    - Product index set $I$: all unique values in column "Product Name"
    - Revenue parameter $A_i$: column "Revenue"
    - Demand parameter $d_i$: column "Demand"
    - Initial Inventory parameter $I_i$: column "Initial Inventory"