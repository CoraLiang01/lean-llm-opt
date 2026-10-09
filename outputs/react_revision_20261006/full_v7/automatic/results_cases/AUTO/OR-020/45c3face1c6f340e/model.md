#### Mathematical Optimization Model

Let:
- $I$ = set of all products, indexed by $i$
- $A_i$ = revenue per unit of product $i$
- $d_i$ = demand for product $i$
- $I_i$ = initial inventory for product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \\
& x_i \leq I_i, && \forall i \in I \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, && \forall i \in I
\end{align*}
\]

#### Data Mapping

- $I$: All unique values in column "Product Name" from table_id file_0_view_0 in SalesDatainBusinesses.csv
- $A_i$: "Revenue" column, table_id file_0_view_0, SalesDatainBusinesses.csv
- $d_i$: "Demand" column, table_id file_0_view_0, SalesDatainBusinesses.csv
- $I_i$: "Initial Inventory" column, table_id file_0_view_0, SalesDatainBusinesses.csv
- $x_i$: Decision variable for each $i \in I$