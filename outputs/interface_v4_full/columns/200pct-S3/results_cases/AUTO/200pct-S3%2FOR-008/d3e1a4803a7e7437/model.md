#### Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from products.csv, column Value)
- $u_i$ = daily inventory limit for vehicle type $i$ (from capacity.csv, column Capacity)
- $C_{tot}$ = total inventory capacity per day (sum of all $u_i$)

**Objective:**
$$
\max \sum_{i \in I} b_i x_i
$$

**Constraints:**
1. **Total Inventory Capacity:**
$$
\sum_{i \in I} x_i \leq C_{tot}
$$

2. **Vehicle Type Inventory Limits:**
$$
x_i \leq u_i \quad \forall i \in I
$$

3. **Integrality and Nonnegativity:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (vehicle types): file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name)
- $b_i$: file_1_view_0.Value, where ProductName = VehicleType $i$
- $u_i$: file_0_view_0.Capacity, where VehicleType = $i$
- $C_{tot}$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity)

All parameters and indices are to be taken directly from the provided CSV files, preserving original order and identifiers.