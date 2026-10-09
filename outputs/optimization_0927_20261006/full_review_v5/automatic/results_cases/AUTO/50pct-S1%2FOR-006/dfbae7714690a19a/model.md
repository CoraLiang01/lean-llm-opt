#### Sets and Indices

Let $i$ index the vehicle types, with the following mapping (in source order):

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

Let $x_i$ be the number of units of vehicle type $i$ to order daily.

#### Parameters

- $v_i$: Value (benefit) of ordering one unit of vehicle type $i$ (see table above).
- $w_i$: Weight (inventory space required) for one unit of vehicle type $i$ (see table above).
- $C$: Total inventory capacity = 1576.

#### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all vehicle types $i$.

#### Mathematical Model

**Objective:**
\[
\max \sum_{i} v_i x_i
\]

**Subject to:**
\[
\sum_{i} w_i x_i \leq 1576
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

#### Parameter Table (source order)

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

**Total Inventory Capacity:** $C = 1576$