Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID from "capacity.csv". The benefit coefficient for each vehicle type is given by the Value in "products.csv", matched by VehicleType/ProductName.

Parameters (in source order):

Vehicle Types, IDs, Per-Type Capacities, and Benefit Coefficients:

\[
\begin{array}{llll}
\text{VehicleID} & \text{VehicleType} & \text{Capacity} & \text{Value} \\
1 & \text{Sedans} & 100 & 1200 \\
2 & \text{SUVs} & 80 & 1800 \\
3 & \text{Electric Vehicles} & 120 & 2500 \\
4 & \text{Hybrid Vehicles} & 90 & 2000 \\
5 & \text{Trucks} & 50 & 1500 \\
6 & \text{Sports Cars} & 30 & 3000 \\
7 & \text{Compact Cars} & 110 & 1000 \\
8 & \text{Luxury Sedans} & 40 & 3500 \\
9 & \text{Vans} & 60 & 1600 \\
10 & \text{Pickup Trucks} & 35 & 1700 \\
\end{array}
\]

Let $x_i$ be the integer number of vehicles of type $i$ to order per day, for $i = 1, \ldots, 10$.

Objective:
\[
\max \; 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

Subject to:

Per-type daily inventory limits:
\[
\begin{align*}
x_1 &\leq 100 \\
x_2 &\leq 80 \\
x_3 &\leq 120 \\
x_4 &\leq 90 \\
x_5 &\leq 50 \\
x_6 &\leq 30 \\
x_7 &\leq 110 \\
x_8 &\leq 40 \\
x_9 &\leq 60 \\
x_{10} &\leq 35 \\
\end{align*}
\]

Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq C_{\text{total}}
\]
where $C_{\text{total}}$ is the total inventory capacity per day (if provided; if not, this constraint is omitted).

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

All parameters and constraints are taken directly from the retrieved data, preserving source order and identifiers.