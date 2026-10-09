### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit for vehicle type $i$

#### Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints:
1. **Vehicle-type daily inventory limits:**
   $$
   x_i \leq u_i \quad \forall i \in I
   $$
2. **Total inventory capacity:**
   $$
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   $$
3. **Integrality and nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

### Data Mapping

- $I$: All vehicle types from `file_0_view_0.VehicleType` and `file_1_view_0.ProductName`
- $b_i$: `file_1_view_0.Value` for vehicle type $i$
- $u_i$: `file_0_view_0.Capacity` for vehicle type $i$
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

**Note:** The mapping between vehicle types in both files is by matching `VehicleType` in `capacity.csv` (`file_0_view_0`) with `ProductName` in `products.csv` (`file_1_view_0`).