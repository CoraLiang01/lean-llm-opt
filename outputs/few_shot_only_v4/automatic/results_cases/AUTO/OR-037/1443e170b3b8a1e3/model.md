Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types listed in the ProductName column.

Parameters:

- $p_i$: Value (profit) of vehicle type $i$
- $w_i$: Weight (space requirement) of vehicle type $i$
- $C$: Total inventory capacity

Data:

From products.csv:

\[
\begin{array}{lll}
\text{ProductName} & p_i & w_i \\
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

From capacity.csv:

\[
C = 765
\]

Model:

Objective:
\[
\max \sum_{i} p_i x_i
\]

Subject to:
\[
\sum_{i} w_i x_i \leq 765
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = profit for vehicle type $i$ (see table above)
- $w_i$ = weight (space requirement) for vehicle type $i$ (see table above)
- $C = 765$ = total inventory capacity

All variables and parameters use the exact identifiers and values as retrieved.