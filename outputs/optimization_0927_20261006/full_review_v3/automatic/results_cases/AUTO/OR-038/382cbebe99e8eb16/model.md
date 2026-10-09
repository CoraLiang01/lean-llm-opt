Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the VehicleID as given below. All $x_i$ are nonnegative integers.

#### Parameters (from retrieved data, in source order):

| VehicleID | VehicleType        | Capacity | Value | Weight |
|-----------|-------------------|----------|-------|--------|
| 1         | Sedans            | 100      | 1200  | 20     |
| 2         | SUVs              | 80       | 1800  | 15     |
| 3         | Electric Vehicles | 120      | 2500  | 25     |
| 4         | Hybrid Vehicles   | 90       | 2000  | 18     |
| 5         | Trucks            | 50       | 1500  | 10     |
| 6         | Sports Cars       | 30       | 3000  | 5      |
| 7         | Compact Cars      | 110      | 1000  | 22     |
| 8         | Luxury Sedans     | 40       | 3500  | 8      |
| 9         | Vans              | 60       | 1600  | 12     |
| 10        | Pickup Trucks     | 35       | 1700  | 7      |

#### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1,2,\ldots,10$

#### Objective Function

$$
\max \; 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

#### Constraints

1. **Vehicle Type Daily Inventory Limits:**
   $$
   x_1 \leq 100
   $$
   $$
   x_2 \leq 80
   $$
   $$
   x_3 \leq 120
   $$
   $$
   x_4 \leq 90
   $$
   $$
   x_5 \leq 50
   $$
   $$
   x_6 \leq 30
   $$
   $$
   x_7 \leq 110
   $$
   $$
   x_8 \leq 40
   $$
   $$
   x_9 \leq 60
   $$
   $$
   x_{10} \leq 35
   $$

2. **Total Inventory Capacity Constraint:**
   
   $$
   x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq C
   $$
   where $C$ is the total inventory capacity per day (must be specified by the user; if not given, this constraint is omitted).

3. **Integrality and Nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
   $$

#### Notes

- $x_i$ is the number of vehicles of type $i$ to order per day.
- The benefit coefficients are given by the "Value" column.
- Each $x_i$ is bounded above by its corresponding "Capacity" from capacity.csv.
- If a total inventory capacity $C$ is specified, include the total inventory constraint as above; otherwise, omit it.

All data and constraints are preserved in source order and with original identifiers.