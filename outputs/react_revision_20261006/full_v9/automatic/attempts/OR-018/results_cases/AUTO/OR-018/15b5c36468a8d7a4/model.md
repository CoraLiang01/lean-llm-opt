#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products classified under ‘Baby’ (from Product Name column, filtered as specified)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from Revenue column)
    - $d_i$ = deterministic demand for product $i$ (parameter, from Demand column)
    - $s_i$ = initial inventory of product $i$ (parameter, from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Objective:
$$
\max \sum_{i \in I} A_i x_i
$$

Subject to:
- Demand and inventory fulfillment bounds:
$$
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
$$
- $x_i \in \mathbb{Z}$ (integer) for all $i \in I$

#### Data Mapping

- Table: file_0_view_0 (from Salesdata.csv)
    - Index set $I$: all rows where Product Name has prefix "Baby"
    - $A_i$: column "Revenue"
    - $d_i$: column "Demand"
    - $s_i$: column "Initial Inventory"
    - $x_i$: decision variable for each $i \in I$