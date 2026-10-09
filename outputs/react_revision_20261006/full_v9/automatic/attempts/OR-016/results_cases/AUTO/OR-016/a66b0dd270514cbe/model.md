#### Mathematical Optimization Model

Let:
- $I$ = set of all products, indexed by $i$ (as defined by all "Product Name" entries in the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = demand for product $i$ (from column "Demand")
    - $s_i$ = initial inventory for product $i$ (from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- Table: file_0_view_0 (from RetailSalesDataset.csv)
    - Index set $I$: All unique values in column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: defined for each $i \in I$ as above

No additional constraints or synthetic scenario parameters are specified in the query. All bounds and coefficients are mapped directly from the provided columns.