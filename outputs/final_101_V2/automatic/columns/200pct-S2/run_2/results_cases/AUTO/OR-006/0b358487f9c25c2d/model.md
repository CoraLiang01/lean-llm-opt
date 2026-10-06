Let $I$ be the set of vehicle types, indexed by $i$, as given by the ProductName column in products.csv. Let $x_i$ be the integer number of units of vehicle type $i$ to order daily.

Parameters:
- $v_i$: Value (benefit) of vehicle type $i$ (from the Value column)
- $w_i$: Weight (inventory space required per unit) of vehicle type $i$ (from the Weight column)
- $C$: Total inventory capacity (from the Capacity column in capacity.csv)

Data:
- Vehicle types and their parameters (in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedan               | 1752  | 15     |
| SUV                 | 1856  | 87     |
| Truck               | 8372  | 36     |
| Convertible         | 6168  | 30     |
| Minivan             | 9681  | 33     |
| Coupe               | 8062  | 72     |
| Hatchback           | 3895  | 75     |
| Station Wagon       | 3254  | 71     |
| Electric Car        | 1701  | 51     |
| Hybrid Car          | 6799  | 21     |
| Luxury Sedan        | 2724  | 97     |
| Sports Car          | 6304  | 52     |
| Crossover           | 3255  | 25     |
| Diesel Truck        | 1923  | 15     |
| Compact SUV         | 4103  | 54     |
| Luxury SUV          | 4429  | 57     |
| Cargo Van           | 2663  | 18     |
| Pickup Truck        | 1691  | 69     |
| Roadster            | 5632  | 26     |
| Muscle Car          | 4793  | 38     |
| Off-road Vehicle    | 1343  | 31     |
| Camper Van          | 9124  | 74     |
| Compact Car         | 3652  | 82     |
| Motorcycle          | 8842  | 49     |
| Electric SUV        | 9176  | 64     |

- Total inventory capacity: $C = 1576$

Mathematical Model:

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq 1576
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Where:
- $v_i$ and $w_i$ are as listed above for each vehicle type $i$ (ProductName, in source order).
- $x_i$ is the integer number of units of vehicle type $i$ to order daily.