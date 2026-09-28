Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ corresponds to the VehicleID in the data below. All $x_i$ are integer and nonnegative.

Maximize total benefit:
$$
\max \; 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
$$

Subject to:

Per-vehicle-type daily inventory limits:
\[
\begin{align*}
x_1 &\leq 100 \quad &\text{(Sedans)} \\
x_2 &\leq 80 \quad &\text{(SUVs)} \\
x_3 &\leq 120 \quad &\text{(Electric Vehicles)} \\
x_4 &\leq 90 \quad &\text{(Hybrid Vehicles)} \\
x_5 &\leq 50 \quad &\text{(Trucks)} \\
x_6 &\leq 30 \quad &\text{(Sports Cars)} \\
x_7 &\leq 110 \quad &\text{(Compact Cars)} \\
x_8 &\leq 40 \quad &\text{(Luxury Sedans)} \\
x_9 &\leq 60 \quad &\text{(Vans)} \\
x_{10} &\leq 35 \quad &\text{(Pickup Trucks)}
\end{align*}
\]

Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq C
\]
where $C$ is the total inventory capacity (to be specified by the dealership; if not given, this constraint can be omitted or set as appropriate).

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

Mapping of VehicleID to VehicleType and Value:
\[
\begin{array}{llll}
\text{VehicleID} & \text{VehicleType} & \text{Value (Benefit)} & \text{Capacity} \\
1 & \text{Sedans} & 1200 & 100 \\
2 & \text{SUVs} & 1800 & 80 \\
3 & \text{Electric Vehicles} & 2500 & 120 \\
4 & \text{Hybrid Vehicles} & 2000 & 90 \\
5 & \text{Trucks} & 1500 & 50 \\
6 & \text{Sports Cars} & 3000 & 30 \\
7 & \text{Compact Cars} & 1000 & 110 \\
8 & \text{Luxury Sedans} & 3500 & 40 \\
9 & \text{Vans} & 1600 & 60 \\
10 & \text{Pickup Trucks} & 1700 & 35 \\
\end{array}
\]

All variables and constraints are included as per the retrieved data and user description.