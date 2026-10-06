#### Abstract Mathematical Model

Let:
- $I$ = set of products classified under ‘ELE-S’, indexed by $i$ (with identifiers Product_Reference from SalesStoreoverview.csv)
- For each $i \in I$:
    - $r_i$ = Revenue per unit of product $i$ (from Revenue column)
    - $d_i$ = Demand for product $i$ (from Demand column)
    - $s_i$ = Initial Inventory of product $i$ (from Initial Inventory column)
- Decision variables: $x_i$ = number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
- Inventory constraint for each product:
$$
x_i \leq s_i \qquad \forall i \in I
$$
- Demand constraint for each product:
$$
x_i \leq d_i \qquad \forall i \in I
$$
- Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

#### Data Mapping

- $I$ (products): All rows in SalesStoreoverview.csv where Product_Reference starts with 'ELE-S'
- $r_i$: SalesStoreoverview.csv, column Revenue, for each $i$
- $d_i$: SalesStoreoverview.csv, column Demand, for each $i$
- $s_i$: SalesStoreoverview.csv, column Initial Inventory, for each $i$
- $x_i$: Decision variable for each $i$ (Product_Reference from SalesStoreoverview.csv)

Each constraint and parameter is mapped directly to the corresponding column and row in the retrieved data (table_id: file_0_view_0). No data is omitted or synthesized.