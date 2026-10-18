Let $I$ be the set of vehicle types:
$$
I = \{\text{Sedan},\ \text{SUV},\ \text{Truck},\ \text{Convertible},\ \text{Minivan},\ \text{Coupe},\ \text{Hatchback},\ \text{Station Wagon},\ \text{Electric Car},\ \text{Hybrid Car},\ \text{Luxury Sedan},\ \text{Sports Car},\ \text{Crossover},\ \text{Diesel Truck},\ \text{Compact SUV},\ \text{Luxury SUV},\ \text{Cargo Van},\ \text{Pickup Truck},\ \text{Roadster},\ \text{Muscle Car},\ \text{Off-road Vehicle},\ \text{Camper Van},\ \text{Compact Car},\ \text{Motorcycle},\ \text{Electric SUV}\}
$$

Let $x_i \geq 0$ (integer or continuous, as appropriate) be the number of vehicles of type $i \in I$ to order per day.

Parameters (from products.csv, in source order):

\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Total inventory capacity (from capacity.csv):

\[
\text{Capacity} = 765
\]

Model:

Maximize total profit:
\[
\max \sum_{i \in I} v_i x_i
\]
where $v_i$ is the Value (profit per unit) for product $i$.

Subject to the total inventory capacity:
\[
\sum_{i \in I} w_i x_i \leq 765
\]
where $w_i$ is the Weight (inventory space per unit) for product $i$.

Non-negativity:
\[
x_i \geq 0 \quad \forall i \in I
\]

Where all coefficients and identifiers are as listed above.