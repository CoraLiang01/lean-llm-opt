## Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from file_0_view_0.resource_id)
- $J$ = set of products, indexed by $j$ (from file_1_view_0.item_name)
- $v_j$ = value of one unit of product $j$ (from file_1_view_0.item_value)
- $a_j$ = space/weight requirement of one unit of product $j$ (from file_1_view_0.resource_requirement)
- $c_i$ = capacity of shelf $i$ (from file_0_view_0.resource_capacity)
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (decision variable, integer, $\geq 0$)

### Objective
Maximize total value of products allocated:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):
   $$
   \sum_{j \in J} a_j \, x_{ij} \leq c_i \quad \forall i \in I
   $$

2. **Nonnegativity and Integrality**:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   $$

---

## Data Mapping

- $I$ (Shelves): file_0_view_0.resource_id
- $J$ (Products): file_1_view_0.item_name
- $v_j$: file_1_view_0.item_value (indexed by item_name)
- $a_j$: file_1_view_0.resource_requirement (indexed by item_name)
- $c_i$: file_0_view_0.resource_capacity (indexed by resource_id)
- $x_{ij}$: Number of units of product $j$ on shelf $i$ (decision variable, integer, $\geq 0$)