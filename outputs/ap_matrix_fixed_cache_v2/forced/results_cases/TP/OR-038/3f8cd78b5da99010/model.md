Let $I$ be the set of vehicle types, indexed in source order as follows:

\[
\begin{array}{llcl}
\text{Index} & \text{VehicleType} & \text{Capacity} & \text{Benefit Coefficient} \\
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

Let $x_i$ be the integer number of vehicles of type $i$ to order per day.

Let $C_i$ be the daily inventory limit (capacity) for vehicle type $i$.

Let $v_i$ be the benefit coefficient for vehicle type $i$.

Let $K = \sum_{i=1}^{10} C_i = 715$ be the total inventory capacity per day.

The model is:

Maximize total benefit:
\[
\max \; 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

Subject to:
\[
\begin{align*}
& x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715 \\
& 0 \leq x_1 \leq 100 \\
& 0 \leq x_2 \leq 80 \\
& 0 \leq x_3 \leq 120 \\
& 0 \leq x_4 \leq 90 \\
& 0 \leq x_5 \leq 50 \\
& 0 \leq x_6 \leq 30 \\
& 0 \leq x_7 \leq 110 \\
& 0 \leq x_8 \leq 40 \\
& 0 \leq x_9 \leq 60 \\
& 0 \leq x_{10} \leq 35 \\
& x_i \in \mathbb{Z}, \quad \forall i=1,\ldots,10
\end{align*}
\]

Where:
- $x_1$: Sedans
- $x_2$: SUVs
- $x_3$: Electric Vehicles
- $x_4$: Hybrid Vehicles
- $x_5$: Trucks
- $x_6$: Sports Cars
- $x_7$: Compact Cars
- $x_8$: Luxury Sedans
- $x_9$: Vans
- $x_{10}$: Pickup Trucks

All coefficients and bounds are as retrieved and in source order.

Retrieved Information

capacity.csv:
1, Sedans, 100
2, SUVs, 80
3, Electric Vehicles, 120
4, Hybrid Vehicles, 90
5, Trucks, 50
6, Sports Cars, 30
7, Compact Cars, 110
8, Luxury Sedans, 40
9, Vans, 60
10, Pickup Trucks, 35

products.csv:
Sedans, 1200
SUVs, 1800
Electric Vehicles, 2500
Hybrid Vehicles, 2000
Trucks, 1500
Sports Cars, 3000
Compact Cars, 1000
Luxury Sedans, 3500
Vans, 1600
Pickup Trucks, 1700