#### Abstract Mathematical Model

Let:

- $I$ = set of products (indexed by $i$)
- $v_i$ = benefit/income per unit of product $i$ (from products.csv, column "Value")
- $w_i$ = weight or stock usage per unit of product $i$ (from products.csv, column "Weight")
- $C$ = overall stock capacity (from capacity.csv, column "Capacity")
- $x_i$ = number of units of product $i$ to order each day (decision variable)

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\]

#### Data Mapping

- Table: capacity.csv, Column: Capacity, Table ID: file_0_view_0 $\rightarrow$ parameter $C$
- Table: products.csv, Column: ProductName, Table ID: file_1_view_0 $\rightarrow$ index set $I$
- Table: products.csv, Column: Value, Table ID: file_1_view_0 $\rightarrow$ parameter $v_i$
- Table: products.csv, Column: Weight, Table ID: file_1_view_0 $\rightarrow$ parameter $w_i$