### Mathematical Model

Let:
- $I$ = set of areas, indexed by $i$ (from ProductName in products.csv)
- $x_i$ = scale of development per day in area $i$ (decision variable, $x_i \geq 0$, integer)
- $v_i$ = development benefit per unit in area $i$ (from Value in products.csv)
- $w_i$ = resource consumption per unit in area $i$ (from Weight in products.csv)
- $C$ = overall development capacity (from Capacity in capacity.csv)

#### Objective:
$\max \sum_{i \in I} v_i x_i$

#### Constraint:
$\sum_{i \in I} w_i x_i \leq C$

#### Variable domains:
$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I$

---

### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (area/ProductName)

All parameters and index sets are mapped directly from the returned CSV data.