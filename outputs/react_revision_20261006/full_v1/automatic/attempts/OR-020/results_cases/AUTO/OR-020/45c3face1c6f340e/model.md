#### Symbolic Mathematical Model

Let:

- $I$ = index set of all products (from "Product Name")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from "Revenue")
    - $d_i$ = demand for product $i$ (from "Demand")
    - $s_i$ = initial inventory for product $i$ (from "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i \quad && \forall i \in I \\
& x_i \leq s_i \quad && \forall i \in I \\
& x_i \geq 0,\ x_i \in \mathbb{Z} \quad && \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from SalesDatainBusinesses.csv)
    - Index set $I$: column "Product Name"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"