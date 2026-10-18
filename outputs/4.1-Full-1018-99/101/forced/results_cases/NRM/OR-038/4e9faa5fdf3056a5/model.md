#### Index Sets

- $I$: set of vehicle types (from products.csv and capacity.csv)

#### Parameters

- $b_i$: benefit coefficient for vehicle type $i \in I$ (from products.csv, column "Value", matched by vehicle type)
- $u_i$: daily inventory limit for vehicle type $i \in I$ (from capacity.csv, column "Capacity")
- $U$: total inventory capacity per day (sum of all $u_i$ from capacity.csv, or as specified if a separate total is provided)

#### Decision Variables

- $x_i$: number of vehicles of type $i$ to order per day, integer, $x_i \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. Vehicle Type Inventory Limits:
   $$
   x_i \leq u_i \quad \forall i \in I
   $$

2. Total Inventory Capacity:
   $$
   \sum_{i \in I} x_i \leq U
   $$

3. Integer and Nonnegativity:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0)
  - VehicleType: vehicle type identifier ($i$)
  - Capacity: daily inventory limit for vehicle type ($u_i$)
- Table: products.csv (table_id: file_1_view_0)
  - ProductName: vehicle type identifier ($i$)
  - Value: benefit coefficient ($b_i$)

Vehicle types are matched between the two tables by VehicleType/ProductName. The total inventory capacity $U$ is the sum of all $u_i$ from capacity.csv unless otherwise specified. All data is used as returned by the query.