Let:
- \( W \) be the set of warehouses, indexed by \( w \), with Warehouse IDs as given in capacity.csv.
- \( P \) be the set of vehicle types, indexed by \( i \), with ProductNames as given in products.csv.
- \( x_{w,i} \) = number of vehicles of type \( i \) to be stored in warehouse \( w \) per day (integer, \( x_{w,i} \geq 0 \)).

Parameters (from the data):
- For each warehouse \( w \):

  | Warehouse ID   | Capacity |
  |----------------|----------|
  | Warehouse 1    | 100      |
  | Warehouse 2    | 80       |
  | Warehouse 3    | 120      |
  | Warehouse 4    | 90       |
  | Warehouse 5    | 50       |
  | Warehouse 6    | 30       |
  | Warehouse 7    | 110      |
  | Warehouse 8    | 40       |
  | Warehouse 9    | 60       |
  | Warehouse 10   | 35       |

- For each vehicle type \( i \):

  | ProductName         | Value | Weight |
  |---------------------|-------|--------|
  | Sedans              | 1200  | 20     |
  | SUVs                | 1800  | 15     |
  | Electric Vehicles   | 2500  | 25     |
  | Hybrid Vehicles     | 2000  | 18     |
  | Trucks              | 1500  | 10     |
  | Sports Cars         | 3000  | 5      |
  | Compact Cars        | 1000  | 22     |
  | Luxury Sedans       | 3500  | 8      |
  | Vans                | 1600  | 12     |
  | Pickup Trucks       | 1700  | 7      |

Model:

Decision variables:
- For each warehouse \( w \) and vehicle type \( i \):
  - \( x_{w,i} \in \mathbb{Z}_+ \) (nonnegative integers)

Objective:
\[
\text{Maximize} \quad Z = \sum_{w \in W} \sum_{i \in P} \text{Value}_i \cdot x_{w,i}
\]
where \(\text{Value}_i\) is the "Value" for vehicle type \(i\) from products.csv.

Subject to, for each warehouse \( w \):
\[
\sum_{i \in P} \text{Weight}_i \cdot x_{w,i} \leq \text{Capacity}_w
\]
where \(\text{Weight}_i\) is the "Weight" for vehicle type \(i\) from products.csv, and \(\text{Capacity}_w\) is the capacity for warehouse \(w\) from capacity.csv.

Variable domains:
\[
x_{w,i} \in \{0, 1, 2, \ldots\} \quad \forall w \in W,\, i \in P
\]

Explicitly, for each warehouse (example for Warehouse 1):
\[
\sum_{i \in P} \text{Weight}_i \cdot x_{\text{Warehouse 1},i} \leq 100
\]
and similarly for each warehouse with its respective capacity.

All variables and constraints are indexed by the explicit Warehouse ID and ProductName as given in the source files. No sorting or re-indexing is performed.

This model maximizes the total value of vehicles stored across all warehouses, subject to each warehouse's capacity, with integer numbers of vehicles per type per warehouse.