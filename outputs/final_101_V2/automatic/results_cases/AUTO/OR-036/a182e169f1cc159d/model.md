Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $x_i \in \mathbb{Z}_{\geq 0}$ for all $i$.

Let the set of vehicle types $i$ and their parameters be:

\[
\begin{array}{lll}
\text{ProductName} & \text{Value}~(v_i) & \text{Weight}~(w_i) \\
\hline
\text{Sedan} & 1752 & 15 \\
\text{SUV} & 1856 & 87 \\
\text{Truck} & 8372 & 36 \\
\text{Convertible} & 6168 & 30 \\
\text{Minivan} & 9681 & 33 \\
\text{Coupe} & 8062 & 72 \\
\text{Hatchback} & 3895 & 75 \\
\text{Station Wagon} & 3254 & 71 \\
\text{Electric Car} & 1701 & 51 \\
\text{Hybrid Car} & 6799 & 21 \\
\text{Luxury Sedan} & 2724 & 97 \\
\text{Sports Car} & 6304 & 52 \\
\text{Crossover} & 3255 & 25 \\
\text{Diesel Truck} & 1923 & 15 \\
\text{Compact SUV} & 4103 & 54 \\
\text{Luxury SUV} & 4429 & 57 \\
\text{Cargo Van} & 2663 & 18 \\
\text{Pickup Truck} & 1691 & 69 \\
\text{Roadster} & 5632 & 26 \\
\text{Muscle Car} & 4793 & 38 \\
\text{Off-road Vehicle} & 1343 & 31 \\
\text{Camper Van} & 9124 & 74 \\
\text{Compact Car} & 3652 & 82 \\
\text{Motorcycle} & 8842 & 49 \\
\text{Electric SUV} & 9176 & 64 \\
\end{array}
\]

Total inventory capacity: $C = 1576$

Objective:
\[
\max \sum_{i} v_i x_i
\]

Subject to:
\[
\sum_{i} w_i x_i \leq 1576
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:
- $v_i$ is the Value for vehicle type $i$ (see table above)
- $w_i$ is the Weight for vehicle type $i$ (see table above)
- $x_i$ is the integer number of units of vehicle type $i$ to order daily

All data and identifiers are as retrieved and preserved in source order.