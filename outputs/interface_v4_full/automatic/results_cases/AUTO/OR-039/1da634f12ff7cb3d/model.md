## Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from products.csv, column ProductName)
- $J$ = set of warehouses, indexed by $j$ (from capacity.csv, column Warehouse ID)

Parameters:
- $v_i$ = value per unit of vehicle type $i$ (from products.csv, column Value, table_id: file_1_view_0)
- $w_i$ = weight (space requirement) per unit of vehicle type $i$ (from products.csv, column Weight, table_id: file_1_view_0)
- $C_j$ = capacity of warehouse $j$ (from capacity.csv, column Capacity, table_id: file_0_view_0)

Decision Variables:
- $x_{ij}$ = number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

### Objective
$$
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
$$

### Constraints

1. **Warehouse Capacity Constraints** (for each warehouse $j$):
$$
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j \quad \forall j \in J
$$

2. **Non-negativity and Integrality**:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, \forall j \in J
$$

---

### Data Mapping

- $I$ (vehicle types): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value
- $w_i$: file_1_view_0, column Weight
- $J$ (warehouses): file_0_view_0, column Warehouse ID
- $C_j$: file_0_view_0, column Capacity

All parameters and indices are to be mapped exactly as in the original files and columns.