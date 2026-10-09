#### Symbolic Mathematical Model

Let:
- $I$ = index set of all baked goods in the bakery (from Product Name column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of baked good $i$ (from Revenue column)
    - $d_i$ = deterministic demand for baked good $i$ (from Demand column)
    - $s_i$ = initial inventory of baked good $i$ (from Initial Inventory column)
    - $x_i$ = quantity of baked good $i$ to fulfill (decision variable)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
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

- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - Index set $I$: Product Name
    - Parameter $A_i$: Revenue
    - Parameter $d_i$: Demand
    - Parameter $s_i$: Initial Inventory

All parameters and index sets are defined from the full set of records in the specified table and columns.