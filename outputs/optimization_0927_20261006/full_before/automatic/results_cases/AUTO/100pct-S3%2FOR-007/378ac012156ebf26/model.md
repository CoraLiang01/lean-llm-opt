Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order given:

Sedan, SUV, Truck, Convertible, Minivan, Coupe, Hatchback, Station Wagon, Electric Car, Hybrid Car, Luxury Sedan, Sports Car, Crossover, Diesel Truck, Compact SUV, Luxury SUV, Cargo Van, Pickup Truck, Roadster, Muscle Car, Off-road Vehicle, Camper Van, Compact Car, Motorcycle, Electric SUV.

Let $v_i$ be the Value (profit per unit) and $w_i$ be the Weight (stock space required per unit) for each product $i$ as given below.

The overall inventory capacity is $765$ units of stock space.

The mathematical model is:

Objective:
$$
\max \sum_{i} v_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 765
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:

\[
\begin{array}{lll}
\text{ProductName} & v_i\ (\text{Value}) & w_i\ (\text{Weight}) \\
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

Capacity:
- Total available stock space: $765$

Decision variables:
- $x_i$: number of vehicles of type $i$ to order per day, integer and nonnegative.

Objective:
- Maximize total profit from all ordered vehicles.

Constraint:
- Total stock space used by all ordered vehicles does not exceed $765$.

Variable domains:
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i$ (integer, nonnegative).