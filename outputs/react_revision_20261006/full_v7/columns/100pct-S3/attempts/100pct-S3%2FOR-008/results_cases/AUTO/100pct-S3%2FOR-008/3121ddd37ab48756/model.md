#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} x_i \leq C
\]
\[
0 \leq x_i \leq u_i \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

#### Data Mapping

- $I$: All vehicle types from file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name)
- $b_i$: file_1_view_0.Value, where ProductName = $i$
- $u_i$: file_0_view_0.Capacity, where VehicleType = $i$
- $C$: $\sum_{i \in I} u_i$ (sum of all file_0_view_0.Capacity)
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

Each $x_i$ is bounded above by its own $u_i$ and the sum of all $x_i$ is bounded by $C$. All parameters are mapped directly from the CSV data as specified.