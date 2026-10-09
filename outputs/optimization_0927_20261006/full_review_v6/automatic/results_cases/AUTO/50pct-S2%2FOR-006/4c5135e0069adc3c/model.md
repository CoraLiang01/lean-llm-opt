Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values:

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

The benefit coefficients $p_i$ and weights $w_i$ for each vehicle type $i$ are as follows:

| ProductName         | Value ($p_i$) | Weight ($w_i$) |
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

The total inventory capacity is:

- Capacity: $1576$

The mathematical model is:

**Objective:**
\[
\max \sum_{i} p_i x_i
\]
where $p_i$ is the Value for product $i$.

**Subject to:**

\[
\sum_{i} w_i x_i \leq 1576
\]
where $w_i$ is the Weight for product $i$.

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $x_i$ = number of units of vehicle type $i$ to order daily (integer, $\geq 0$)
- $p_i$ = Value for vehicle type $i$ (see table above)
- $w_i$ = Weight for vehicle type $i$ (see table above)

All data and identifiers are as retrieved and preserved in source order.