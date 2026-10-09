## Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$, with StorageIDs from file_0_view_0["StorageID"]
- $J$ = set of air conditioner types, indexed by $j$, with ProductNames from file_1_view_0["ProductName"]

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0["Capacity"])
- $v_j$ = value per unit of air conditioner type $j$ (file_1_view_0["Value"])
- $w_j$ = size (weight) per unit of air conditioner type $j$ (file_1_view_0["Weight"])

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

### Objective
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

### Constraints

1. **Storage Area Capacity Constraints** (for all $i \in I$):
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i
$$

2. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

## Data Mapping

- $I$ (storage areas): file_0_view_0["StorageID"]
- $J$ (air conditioner types): file_1_view_0["ProductName"]
- $c_i$: file_0_view_0["Capacity"], keyed by StorageID
- $v_j$: file_1_view_0["Value"], keyed by ProductName
- $w_j$: file_1_view_0["Weight"], keyed by ProductName
- $x_{ij}$: number of units of product $j$ in storage area $i$ (decision variable, integer, $\geq 0$)

All parameters are mapped directly from the corresponding columns in the returned CSV files, using the business identifiers StorageID and ProductName.