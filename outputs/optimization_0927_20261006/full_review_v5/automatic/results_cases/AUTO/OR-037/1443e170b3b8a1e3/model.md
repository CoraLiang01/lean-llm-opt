#### Sets and Indices

Let $i$ index the vehicle types as listed in the table below.

#### Parameters

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedan               | 2524  | 99     |
| SUV                 | 4614  | 55     |
| Truck               | 8416  | 75     |
| Convertible         | 5917  | 94     |
| Minivan             | 9048  | 80     |
| Coupe               | 1140  | 82     |
| Hatchback           | 8962  | 71     |
| Station Wagon       | 1888  | 100    |
| Electric Car        | 8487  | 28     |
| Hybrid Car          | 4425  | 93     |
| Luxury Sedan        | 4717  | 84     |
| Sports Car          | 4210  | 83     |
| Crossover           | 1226  | 62     |
| Diesel Truck        | 7400  | 90     |
| Compact SUV         | 4639  | 99     |
| Luxury SUV          | 7712  | 96     |
| Cargo Van           | 3299  | 21     |
| Pickup Truck        | 9895  | 39     |
| Roadster            | 4496  | 99     |
| Muscle Car          | 4526  | 81     |
| Off-road Vehicle    | 5688  | 6      |
| Camper Van          | 3007  | 58     |
| Compact Car         | 3623  | 37     |
| Motorcycle          | 8474  | 15     |
| Electric SUV        | 8372  | 37     |

Total inventory capacity: $765$

Let $p_i$ = Value for product $i$ (profit per unit)  
Let $w_i$ = Weight for product $i$ (space per unit)  
Let $C = 765$ (total inventory capacity)

#### Decision Variables

$x_i$ = number of vehicles of type $i$ to order per day  
Domain: $x_i \in \mathbb{Z}_{\geq 0}$ (nonnegative integers)

#### Objective

Maximize total profit:
$$
\max \sum_{i} p_i x_i
$$

#### Constraints

Inventory capacity constraint:
$$
\sum_{i} w_i x_i \leq 765
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

#### Parameter Table

| ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------|---------------|---------------|
| Sedan               | 2524          | 99            |
| SUV                 | 4614          | 55            |
| Truck               | 8416          | 75            |
| Convertible         | 5917          | 94            |
| Minivan             | 9048          | 80            |
| Coupe               | 1140          | 82            |
| Hatchback           | 8962          | 71            |
| Station Wagon       | 1888          | 100           |
| Electric Car        | 8487          | 28            |
| Hybrid Car          | 4425          | 93            |
| Luxury Sedan        | 4717          | 84            |
| Sports Car          | 4210          | 83            |
| Crossover           | 1226          | 62            |
| Diesel Truck        | 7400          | 90            |
| Compact SUV         | 4639          | 99            |
| Luxury SUV          | 7712          | 96            |
| Cargo Van           | 3299          | 21            |
| Pickup Truck        | 9895          | 39            |
| Roadster            | 4496          | 99            |
| Muscle Car          | 4526          | 81            |
| Off-road Vehicle    | 5688          | 6             |
| Camper Van          | 3007          | 58            |
| Compact Car         | 3623          | 37            |
| Motorcycle          | 8474          | 15            |
| Electric SUV        | 8372          | 37            |

#### Capacity

$C = 765$

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i} p_i x_i \\
\text{s.t.} \quad & \sum_{i} w_i x_i \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

where $i$ runs over all ProductName as listed above, with $p_i$ and $w_i$ as given.