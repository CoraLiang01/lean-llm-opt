##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day, for each $i \in I$ (where $I$ is the set of all vehicle types listed below).

##### Parameters

Let $I$ be the set of vehicle types:
- Sedan
- SUV
- Truck
- Convertible
- Minivan
- Coupe
- Hatchback
- Station Wagon
- Electric Car
- Hybrid Car
- Luxury Sedan
- Sports Car
- Crossover
- Diesel Truck
- Compact SUV
- Luxury SUV
- Cargo Van
- Pickup Truck
- Roadster
- Muscle Car
- Off-road Vehicle
- Camper Van
- Compact Car
- Motorcycle
- Electric SUV

For each $i \in I$:
- $v_i$: Profit per unit of vehicle type $i$
- $w_i$: Inventory weight per unit of vehicle type $i$

Parameter values (from products.csv):

| Vehicle Type        | $v_i$ (Profit) | $w_i$ (Weight) |
|---------------------|:--------------:|:--------------:|
| Sedan               | 2524           | 99             |
| SUV                 | 4614           | 55             |
| Truck               | 8416           | 75             |
| Convertible         | 5917           | 94             |
| Minivan             | 9048           | 80             |
| Coupe               | 1140           | 82             |
| Hatchback           | 8962           | 71             |
| Station Wagon       | 1888           | 100            |
| Electric Car        | 8487           | 28             |
| Hybrid Car          | 4425           | 93             |
| Luxury Sedan        | 4717           | 84             |
| Sports Car          | 4210           | 83             |
| Crossover           | 1226           | 62             |
| Diesel Truck        | 7400           | 90             |
| Compact SUV         | 4639           | 99             |
| Luxury SUV          | 7712           | 96             |
| Cargo Van           | 3299           | 21             |
| Pickup Truck        | 9895           | 39             |
| Roadster            | 4496           | 99             |
| Muscle Car          | 4526           | 81             |
| Off-road Vehicle    | 5688           | 6              |
| Camper Van          | 3007           | 58             |
| Compact Car         | 3623           | 37             |
| Motorcycle          | 8474           | 15             |
| Electric SUV        | 8372           | 37             |

Total inventory capacity (from capacity.csv): $C = 765$

##### Objective Function

\[
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

1. Inventory capacity constraint:
   \[
   \sum_{i \in I} w_i x_i \leq 765
   \]
2. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} v_i x_i \\
\text{s.t.} \quad & \sum_{i \in I} w_i x_i \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

Where all $v_i$ and $w_i$ are as listed above, and $I$ is the set of all 25 vehicle types.