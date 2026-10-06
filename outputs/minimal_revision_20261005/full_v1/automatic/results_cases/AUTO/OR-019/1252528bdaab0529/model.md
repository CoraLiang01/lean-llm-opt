#### Symbolic Optimization Model

Let:
- $I$ = index set of all products classified under ‘27in’ (from Product Name column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from Revenue column)
    - $d_i$ = demand for product $i$ (from Demand column)
    - $s_i$ = initial inventory for product $i$ (from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

Subject to:
- Inventory and demand bounds:
$$
0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I
$$

- Variable domain:
$$
x_i \in \mathbb{Z} \quad \forall i \in I
$$

#### Data Mapping

- Table: SalesDataAnalysis.csv
    - Index set $I$: All rows where [Product Name] starts with ‘27in’
    - Parameter $A_i$: [Revenue] column
    - Parameter $d_i$: [Demand] column
    - Parameter $s_i$: [Initial Inventory] column
    - Variable $x_i$: Decision variable for each $i \in I$