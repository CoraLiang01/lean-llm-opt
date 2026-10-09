#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$
- $A_i$ = revenue per unit of product $i$
- $d_i$ = deterministic demand for product $i$
- $s_i$ = initial inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All product identifiers from table_id file_0_view_0, column "Product Name"
- $A_i$: Revenue per unit from table_id file_0_view_0, column "Revenue"
- $d_i$: Demand from table_id file_0_view_0, column "Demand"
- $s_i$: Initial Inventory from table_id file_0_view_0, column "Initial Inventory"