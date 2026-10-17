Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following VehicleTypes (with VehicleID as business key):

\[
\begin{array}{lll}
\text{VehicleID} & \text{VehicleType} & \text{Value} \\
1 & \text{Sedans} & 1200 \\
2 & \text{SUVs} & 1800 \\
3 & \text{Electric Vehicles} & 2500 \\
4 & \text{Hybrid Vehicles} & 2000 \\
5 & \text{Trucks} & 1500 \\
6 & \text{Sports Cars} & 3000 \\
7 & \text{Compact Cars} & 1000 \\
8 & \text{Luxury Sedans} & 3500 \\
9 & \text{Vans} & 1600 \\
10 & \text{Pickup Trucks} & 1700 \\
\end{array}
\]

Parameters:
- $b_i$: Value (benefit coefficient) for vehicle type $i$ (from products.csv)
- $u_i$: Per-type daily inventory limit for vehicle type $i$ (from capacity.csv, column "Capacity")
- $x_i$: Integer variable, number of vehicles of type $i$ to order per day

Let $I = \{1,2,\ldots,10\}$ index the VehicleIDs in source order.

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day.

The model is:

\[
\textbf{Maximize} \quad Z = \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq u_i, && \forall i \in I \\
& \sum_{i \in I} x_i \leq C \\
& x_i \in \mathbb{Z}_{\geq 0}, && \forall i \in I
\end{align*}
\]

Where:

\[
\begin{array}{llll}
i & \text{VehicleType} & b_i & u_i \\
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

That is,

\[
\begin{align*}
\text{Maximize} \quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{Subject to:} \quad
& x_1 \leq 100 \\
& x_2 \leq 80 \\
& x_3 \leq 120 \\
& x_4 \leq 90 \\
& x_5 \leq 50 \\
& x_6 \leq 30 \\
& x_7 \leq 110 \\
& x_8 \leq 40 \\
& x_9 \leq 60 \\
& x_{10} \leq 35 \\
& x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where the total inventory capacity per day is $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$.

All identifiers and coefficients are preserved in source order.