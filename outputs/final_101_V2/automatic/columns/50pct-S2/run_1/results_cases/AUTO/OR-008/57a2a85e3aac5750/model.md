Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the vehicle types as listed in the data.

Objective:
$$
\max\ 1200\,x_{\text{Sedans}} + 1800\,x_{\text{SUVs}} + 2500\,x_{\text{Electric Vehicles}} + 2000\,x_{\text{Hybrid Vehicles}} + 1500\,x_{\text{Trucks}} + 3000\,x_{\text{Sports Cars}} + 1000\,x_{\text{Compact Cars}} + 3500\,x_{\text{Luxury Sedans}} + 1600\,x_{\text{Vans}} + 1700\,x_{\text{Pickup Trucks}}
$$

Subject to:

Per-vehicle-type daily inventory limits:
\[
\begin{align*}
x_{\text{Sedans}} &\leq 100 \\
x_{\text{SUVs}} &\leq 80 \\
x_{\text{Electric Vehicles}} &\leq 120 \\
x_{\text{Hybrid Vehicles}} &\leq 90 \\
x_{\text{Trucks}} &\leq 50 \\
x_{\text{Sports Cars}} &\leq 30 \\
x_{\text{Compact Cars}} &\leq 110 \\
x_{\text{Luxury Sedans}} &\leq 40 \\
x_{\text{Vans}} &\leq 60 \\
x_{\text{Pickup Trucks}} &\leq 35 \\
\end{align*}
\]

Total inventory capacity constraint:
\[
x_{\text{Sedans}} + x_{\text{SUVs}} + x_{\text{Electric Vehicles}} + x_{\text{Hybrid Vehicles}} + x_{\text{Trucks}} + x_{\text{Sports Cars}} + x_{\text{Compact Cars}} + x_{\text{Luxury Sedans}} + x_{\text{Vans}} + x_{\text{Pickup Trucks}} \leq C
\]
where $C$ is the total inventory capacity per day (if provided; if not, omit or set as needed).

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

- Vehicle types and their benefit coefficients (from products.csv):

| ProductName         | Value |
|---------------------|-------|
| Sedans              | 1200  |
| SUVs                | 1800  |
| Electric Vehicles   | 2500  |
| Hybrid Vehicles     | 2000  |
| Trucks              | 1500  |
| Sports Cars         | 3000  |
| Compact Cars        | 1000  |
| Luxury Sedans       | 3500  |
| Vans                | 1600  |
| Pickup Trucks       | 1700  |

- Per-type daily inventory limits (from capacity.csv):

| VehicleType        | Capacity |
|--------------------|----------|
| Sedans             | 100      |
| SUVs               | 80       |
| Electric Vehicles  | 120      |
| Hybrid Vehicles    | 90       |
| Trucks             | 50       |
| Sports Cars        | 30       |
| Compact Cars       | 110      |
| Luxury Sedans      | 40       |
| Vans               | 60       |
| Pickup Trucks      | 35       |

All $x_i$ are nonnegative integers.