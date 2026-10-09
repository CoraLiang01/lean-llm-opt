### Mathematical Model

Let:
- $I$ = set of areas (from products.csv, column ProductName)
- For each $i \in I$:
    - $v_i$ = benefit coefficient for area $i$ (from products.csv, column Value)
    - $w_i$ = development unit weight for area $i$ (from products.csv, column Weight)
- $C$ = overall development capacity (from capacity.csv, column Capacity)
- Decision variables: $x_i$ = integer number of development units in area $i$ per day

#### Objective:
$\max \sum_{i \in I} v_i x_i$

#### Constraint:
$\sum_{i \in I} w_i x_i \leq C$

#### Variable domains:
$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I$

---

### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0 (products.csv), keyed by ProductName
- $w_i$: Weight column in file_1_view_0 (products.csv), keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)