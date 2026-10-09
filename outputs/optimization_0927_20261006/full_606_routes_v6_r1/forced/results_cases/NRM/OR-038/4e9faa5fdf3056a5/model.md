#### Index Sets

- $I$: Set of vehicle types (from products.csv and capacity.csv).

#### Parameters

- $b_i$: Benefit coefficient for vehicle type $i \in I$ (from products.csv, column "Value", matched by vehicle type).
- $u_i$: Daily inventory limit (capacity) for vehicle type $i \in I$ (from capacity.csv, column "Capacity").
- $C$: Total inventory capacity per day (sum of all $u_i$ or as specified in capacity.csv if present as a separate field).

#### Decision Variables

- $x_i$: Number of vehicles of type $i$ to order per day, integer, $x_i \in \mathbb{Z}_{\geq 0}$.

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
   \sum_{i \in I} x_i \leq C
   $$

3. Integer and Nonnegativity:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0)
  - Vehicle type: column "VehicleType"
  - Daily inventory limit: column "Capacity"
- Table: products.csv (table_id: file_1_view_0)
  - Vehicle type: column "ProductName"
  - Benefit coefficient: column "Value"

Vehicle types are matched between "VehicleType" in capacity.csv and "ProductName" in products.csv. All records from both tables are used. The total inventory capacity $C$ is the sum of all "Capacity" values in capacity.csv unless otherwise specified.