### Mathematical Model

Let:
- $I$ = set of areas (from ProductName in file_1_view_0)
- For each $i \in I$:
    - $v_i$ = Value of developing area $i$ (from Value in file_1_view_0)
    - $w_i$ = Weight (resource consumption) for area $i$ (from Weight in file_1_view_0)
    - $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $C$ = overall development capacity (from Capacity in file_0_view_0)

#### Objective:
$\max \sum_{i \in I} v_i x_i$

#### Constraint:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I$

---

### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)