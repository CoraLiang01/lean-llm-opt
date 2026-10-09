### Mathematical Model

Let:
- $I$ = set of areas, indexed by $i$, with area names from file_1_view_0.ProductName
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $v_i$ = development benefit of area $i$ (file_1_view_0.Value)
- $w_i$ = development capacity required for area $i$ (file_1_view_0.Weight)
- $C$ = overall development capacity limit (file_0_view_0.Capacity)

#### Objective:
$\max \sum_{i \in I} v_i x_i$

#### Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

### Data Mapping

- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, mapped by ProductName
- $w_i$: file_1_view_0.Weight, mapped by ProductName
- $C$: file_0_view_0.Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, nonnegative integer, indexed by file_1_view_0.ProductName)