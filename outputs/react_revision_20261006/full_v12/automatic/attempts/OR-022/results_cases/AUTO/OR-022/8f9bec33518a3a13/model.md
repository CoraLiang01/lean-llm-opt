#### Symbolic Mathematical Model

Let $I$ be the set of all products in the dataset whose "Product Name" contains "27in".

**Index Set:**
- $I$ = set of all products with "27in" in "Product Name" (from file_0_view_0, column "Product Name")

**Parameters:**
- $A_i$ = revenue per unit of product $i$ (from file_0_view_0, column "Revenue")
- $d_i$ = demand for product $i$ (from file_0_view_0, column "Demand")
- $s_i$ = initial inventory of product $i$ (from file_0_view_0, column "Initial Inventory")

**Decision Variables:**
- $x_i$ = number of units of product $i$ to fulfill, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
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

- Source table: file_0_view_0 (Salesorders.csv)
- Index set $I$: All rows where "Product Name" contains "27in"
- Parameter $A_i$: column "Revenue"
- Parameter $d_i$: column "Demand"
- Parameter $s_i$: column "Initial Inventory"
- Decision variable $x_i$: number of units fulfilled for each $i \in I$