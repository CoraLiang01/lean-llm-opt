### Mathematical Model

Let $I$ be the set of vehicle types (indexed by $i$), with parameters and identifiers as in file_1_view_0[ProductName]. Let $x_i$ be the number of units of vehicle type $i$ to order daily.

**Parameters:**
- $v_i$: Value (benefit) of vehicle type $i$ (file_1_view_0[Value])
- $w_i$: Weight (inventory space required) of vehicle type $i$ (file_1_view_0[Weight])
- $C$: Total inventory capacity (file_0_view_0[Capacity])

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ to order daily

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All records in file_1_view_0[ProductName]
- $v_i$: file_1_view_0[Value] for vehicle $i$
- $w_i$: file_1_view_0[Weight] for vehicle $i$
- $C$: file_0_view_0[Capacity]
- $x_i$: Number of units of vehicle $i$ to order daily (decision variable, integer, nonnegative)