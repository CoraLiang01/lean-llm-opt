#### Symbolic Mathematical Model

Let:

- $I$ = set of all pizza types (from column "Product Name" in table_id file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of pizza type $i$ (from "Revenue")
    - $d_i$ = total demand for pizza type $i$ (from "Demand")
    - $s_i$ = initial inventory for pizza type $i$ (from "Initial Inventory")
    - $x_i$ = number of units of pizza type $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All unique values in column "Product Name" from table_id file_0_view_0 (PizzaSalesDataset.csv)
- Parameter $A_i$: "Revenue" column, table_id file_0_view_0
- Parameter $d_i$: "Demand" column, table_id file_0_view_0
- Parameter $s_i$: "Initial Inventory" column, table_id file_0_view_0
- Variable $x_i$: fulfillment quantity for pizza type $i$ (decision variable, non-negative integer)