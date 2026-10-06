Let $I$ be the set of vehicle types, indexed by $i$, with ProductName and associated parameters as given in products.csv. Let $x_i$ be the integer number of units of vehicle type $i$ to order daily.

Parameters (from products.csv):

- $v_i$: Value (benefit) of vehicle type $i$
- $w_i$: Weight of vehicle type $i$

From capacity.csv:

- $C$: Total inventory capacity = 1576

The mathematical model is:

$$
\begin{align*}
\max \quad & \sum_{i \in I} v_i x_i \\
\text{s.t.} \quad & \sum_{i \in I} w_i x_i \leq 1576 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$

Where:

\[
\begin{array}{ll}
\text{ProductName} & (v_i, w_i) \\
\hline
\text{Sedan} & (1752, 15) \\
\text{SUV} & (1856, 87) \\
\text{Truck} & (8372, 36) \\
\text{Convertible} & (6168, 30) \\
\text{Minivan} & (9681, 33) \\
\text{Coupe} & (8062, 72) \\
\text{Hatchback} & (3895, 75) \\
\text{Station Wagon} & (3254, 71) \\
\text{Electric Car} & (1701, 51) \\
\text{Hybrid Car} & (6799, 21) \\
\text{Luxury Sedan} & (2724, 97) \\
\text{Sports Car} & (6304, 52) \\
\text{Crossover} & (3255, 25) \\
\text{Diesel Truck} & (1923, 15) \\
\text{Compact SUV} & (4103, 54) \\
\text{Luxury SUV} & (4429, 57) \\
\text{Cargo Van} & (2663, 18) \\
\text{Pickup Truck} & (1691, 69) \\
\text{Roadster} & (5632, 26) \\
\text{Muscle Car} & (4793, 38) \\
\text{Off-road Vehicle} & (1343, 31) \\
\text{Camper Van} & (9124, 74) \\
\text{Compact Car} & (3652, 82) \\
\text{Motorcycle} & (8842, 49) \\
\text{Electric SUV} & (9176, 64) \\
\end{array}
\]

Decision variables:

\[
x_i = \text{number of units of vehicle type } i \text{ to order daily}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]

Objective:

\[
\max \left(1752x_{\text{Sedan}} + 1856x_{\text{SUV}} + 8372x_{\text{Truck}} + 6168x_{\text{Convertible}} + 9681x_{\text{Minivan}} + 8062x_{\text{Coupe}} + 3895x_{\text{Hatchback}} + 3254x_{\text{Station Wagon}} + 1701x_{\text{Electric Car}} + 6799x_{\text{Hybrid Car}} + 2724x_{\text{Luxury Sedan}} + 6304x_{\text{Sports Car}} + 3255x_{\text{Crossover}} + 1923x_{\text{Diesel Truck}} + 4103x_{\text{Compact SUV}} + 4429x_{\text{Luxury SUV}} + 2663x_{\text{Cargo Van}} + 1691x_{\text{Pickup Truck}} + 5632x_{\text{Roadster}} + 4793x_{\text{Muscle Car}} + 1343x_{\text{Off-road Vehicle}} + 9124x_{\text{Camper Van}} + 3652x_{\text{Compact Car}} + 8842x_{\text{Motorcycle}} + 9176x_{\text{Electric SUV}} \right)
\]

Subject to:

\[
15x_{\text{Sedan}} + 87x_{\text{SUV}} + 36x_{\text{Truck}} + 30x_{\text{Convertible}} + 33x_{\text{Minivan}} + 72x_{\text{Coupe}} + 75x_{\text{Hatchback}} + 71x_{\text{Station Wagon}} + 51x_{\text{Electric Car}} + 21x_{\text{Hybrid Car}} + 97x_{\text{Luxury Sedan}} + 52x_{\text{Sports Car}} + 25x_{\text{Crossover}} + 15x_{\text{Diesel Truck}} + 54x_{\text{Compact SUV}} + 57x_{\text{Luxury SUV}} + 18x_{\text{Cargo Van}} + 69x_{\text{Pickup Truck}} + 26x_{\text{Roadster}} + 38x_{\text{Muscle Car}} + 31x_{\text{Off-road Vehicle}} + 74x_{\text{Camper Van}} + 82x_{\text{Compact Car}} + 49x_{\text{Motorcycle}} + 64x_{\text{Electric SUV}} \leq 1576
\]

and

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]