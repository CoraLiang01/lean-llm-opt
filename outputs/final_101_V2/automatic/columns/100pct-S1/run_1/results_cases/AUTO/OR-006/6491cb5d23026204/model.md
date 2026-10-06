Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the vehicle types as identified by the ProductName column in products.csv. All $x_i$ are integer and nonnegative.

Parameters (from products.csv and capacity.csv):

- For each vehicle type $i$:
    - $p_i$: Value (benefit coefficient) of vehicle type $i$
    - $w_i$: Weight (inventory space required per unit) of vehicle type $i$
- $C$: Total inventory capacity (from capacity.csv)

Vehicle types and coefficients:

| ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------|--------------|---------------|
| Sedan               | 1752         | 15            |
| SUV                 | 1856         | 87            |
| Truck               | 8372         | 36            |
| Convertible         | 6168         | 30            |
| Minivan             | 9681         | 33            |
| Coupe               | 8062         | 72            |
| Hatchback           | 3895         | 75            |
| Station Wagon       | 3254         | 71            |
| Electric Car        | 1701         | 51            |
| Hybrid Car          | 6799         | 21            |
| Luxury Sedan        | 2724         | 97            |
| Sports Car          | 6304         | 52            |
| Crossover           | 3255         | 25            |
| Diesel Truck        | 1923         | 15            |
| Compact SUV         | 4103         | 54            |
| Luxury SUV          | 4429         | 57            |
| Cargo Van           | 2663         | 18            |
| Pickup Truck        | 1691         | 69            |
| Roadster            | 5632         | 26            |
| Muscle Car          | 4793         | 38            |
| Off-road Vehicle    | 1343         | 31            |
| Camper Van          | 9124         | 74            |
| Compact Car         | 3652         | 82            |
| Motorcycle          | 8842         | 49            |
| Electric SUV        | 9176         | 64            |

Total inventory capacity: $C = 1576$

Mathematical Model:

Objective:
$$
\max \sum_{i} p_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 1576
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{Sedan, SUV, Truck, Convertible, Minivan, Coupe, Hatchback, Station Wagon, Electric Car, Hybrid Car, Luxury Sedan, Sports Car, Crossover, Diesel Truck, Compact SUV, Luxury SUV, Cargo Van, Pickup Truck, Roadster, Muscle Car, Off-road Vehicle, Camper Van, Compact Car, Motorcycle, Electric SUV}\}
$$

Where:
- $p_i$ and $w_i$ are as listed above for each vehicle type $i$.
- $x_i$ is the integer number of units of vehicle type $i$ to order daily.