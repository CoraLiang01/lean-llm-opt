#### Index Sets

- $C$: set of cabinets (indexed by $i$), from table_id: file_0_view_0, column: CabinetID
- $P$: set of coffee products (indexed by $j$), from table_id: file_1_view_0, column: ProductName

#### Parameters

- $cap_i$: capacity of cabinet $i$, from table_id: file_0_view_0, column: Capacity
- $v_j$: value per unit of product $j$, from table_id: file_1_view_0, column: Value
- $w_j$: weight per unit of product $j$, from table_id: file_1_view_0, column: Weight

#### Decision Variables

- $x_{ij}$: number of units of product $j$ to place in cabinet $i$, integer, $x_{ij} \geq 0$

#### Objective

$$
\max \sum_{i \in C} \sum_{j \in P} v_j \cdot x_{ij}
$$

#### Constraints

1. **Cabinet Capacity Constraints** (for all $i \in C$):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq cap_i
$$

2. **Non-negativity and Integrality** (for all $i \in C$, $j \in P$):

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data Mapping

- Cabinets and their capacities: table_id: file_0_view_0, columns: CabinetID, Capacity (from "capacity.csv")
- Products, their values, and weights: table_id: file_1_view_0, columns: ProductName, Value, Weight (from "products.csv")