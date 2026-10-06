Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types as given in the original order:

\[
\begin{array}{ll}
1: & \text{Sedan} \\
2: & \text{SUV} \\
3: & \text{Truck} \\
4: & \text{Convertible} \\
5: & \text{Minivan} \\
6: & \text{Coupe} \\
7: & \text{Hatchback} \\
8: & \text{Station Wagon} \\
9: & \text{Electric Car} \\
10: & \text{Hybrid Car} \\
11: & \text{Luxury Sedan} \\
12: & \text{Sports Car} \\
13: & \text{Crossover} \\
14: & \text{Diesel Truck} \\
15: & \text{Compact SUV} \\
16: & \text{Luxury SUV} \\
17: & \text{Cargo Van} \\
18: & \text{Pickup Truck} \\
19: & \text{Roadster} \\
20: & \text{Muscle Car} \\
21: & \text{Off-road Vehicle} \\
22: & \text{Camper Van} \\
23: & \text{Compact Car} \\
24: & \text{Motorcycle} \\
25: & \text{Electric SUV} \\
\end{array}
\]

Let $p_i$ be the profit (Value) per unit of vehicle $i$, and $w_i$ be the weight (stock space required) per unit of vehicle $i$, as given below (in source order):

\[
\begin{array}{lll}
\text{ProductName} & p_i & w_i \\
\hline
\text{Sedan} & 2524 & 99 \\
\text{SUV} & 4614 & 55 \\
\text{Truck} & 8416 & 75 \\
\text{Convertible} & 5917 & 94 \\
\text{Minivan} & 9048 & 80 \\
\text{Coupe} & 1140 & 82 \\
\text{Hatchback} & 8962 & 71 \\
\text{Station Wagon} & 1888 & 100 \\
\text{Electric Car} & 8487 & 28 \\
\text{Hybrid Car} & 4425 & 93 \\
\text{Luxury Sedan} & 4717 & 84 \\
\text{Sports Car} & 4210 & 83 \\
\text{Crossover} & 1226 & 62 \\
\text{Diesel Truck} & 7400 & 90 \\
\text{Compact SUV} & 4639 & 99 \\
\text{Luxury SUV} & 7712 & 96 \\
\text{Cargo Van} & 3299 & 21 \\
\text{Pickup Truck} & 9895 & 39 \\
\text{Roadster} & 4496 & 99 \\
\text{Muscle Car} & 4526 & 81 \\
\text{Off-road Vehicle} & 5688 & 6 \\
\text{Camper Van} & 3007 & 58 \\
\text{Compact Car} & 3623 & 37 \\
\text{Motorcycle} & 8474 & 15 \\
\text{Electric SUV} & 8372 & 37 \\
\end{array}
\]

Let $C$ be the overall inventory capacity (from capacity.csv; value not shown in the data above, so must be supplied in the actual file).

The mathematical model is:

Objective:
\[
\max \sum_{i=1}^{25} p_i x_i
\]

Subject to:
\[
\sum_{i=1}^{25} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

where:
- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = profit per unit of vehicle $i$ (see table above)
- $w_i$ = weight (stock space required) per unit of vehicle $i$ (see table above)
- $C$ = overall inventory capacity (from capacity.csv)