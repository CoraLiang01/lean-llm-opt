#### Abstract Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from file_1_view_0[ProductName]
- $x_i$ = number of units of vehicle type $i$ to order daily (integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from file_1_view_0[Value])
- $w_i$ = weight (inventory space requirement) for vehicle type $i$ (from file_1_view_0[Weight])
- $C$ = total inventory capacity (from file_0_view_0[Capacity])

**Objective:**
$$
\max \sum_{i \in I} b_i x_i
$$

**Subject to:**
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (vehicle types): file_1_view_0[ProductName]
- $b_i$: file_1_view_0[ProductName, Value]
- $w_i$: file_1_view_0[ProductName, Weight]
- $C$: file_0_view_0[Capacity] (single value)

---

**Decision variables $x_i$ are nonnegative integers representing the number of units of each vehicle type to order daily.**