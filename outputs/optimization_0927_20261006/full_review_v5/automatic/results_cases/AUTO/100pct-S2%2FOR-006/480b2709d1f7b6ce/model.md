Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values from products.csv.

#### Sets and Parameters

- Let $I$ be the set of vehicle types (ProductName):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Sedan                 | 1752  | 15     |
| SUV                   | 1856  | 87     |
| Truck                 | 8372  | 36     |
| Convertible           | 6168  | 30     |
| Minivan               | 9681  | 33     |
| Coupe                 | 8062  | 72     |
| Hatchback             | 3895  | 75     |
| Station Wagon         | 3254  | 71     |
| Electric Car          | 1701  | 51     |
| Hybrid Car            | 6799  | 21     |
| Luxury Sedan          | 2724  | 97     |
| Sports Car            | 6304  | 52     |
| Crossover             | 3255  | 25     |
| Diesel Truck          | 1923  | 15     |
| Compact SUV           | 4103  | 54     |
| Luxury SUV            | 4429  | 57     |
| Cargo Van             | 2663  | 18     |
| Pickup Truck          | 1691  | 69     |
| Roadster              | 5632  | 26     |
| Muscle Car            | 4793  | 38     |
| Off-road Vehicle      | 1343  | 31     |
| Camper Van            | 9124  | 74     |
| Compact Car           | 3652  | 82     |
| Motorcycle            | 8842  | 49     |
| Electric SUV          | 9176  | 64     |

- Let $v_i$ be the Value for vehicle type $i$.
- Let $w_i$ be the Weight for vehicle type $i$.
- The total inventory capacity is $C = 1576$.

#### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of units of vehicle type $i$ to order daily).

#### Objective

Maximize total benefit:
$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraint

Total inventory weight cannot exceed capacity:
$$
\sum_{i \in I} w_i x_i \leq 1576
$$

#### Variable Domains

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

#### Parameter Table

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Sedan                 | 1752  | 15     |
| SUV                   | 1856  | 87     |
| Truck                 | 8372  | 36     |
| Convertible           | 6168  | 30     |
| Minivan               | 9681  | 33     |
| Coupe                 | 8062  | 72     |
| Hatchback             | 3895  | 75     |
| Station Wagon         | 3254  | 71     |
| Electric Car          | 1701  | 51     |
| Hybrid Car            | 6799  | 21     |
| Luxury Sedan          | 2724  | 97     |
| Sports Car            | 6304  | 52     |
| Crossover             | 3255  | 25     |
| Diesel Truck          | 1923  | 15     |
| Compact SUV           | 4103  | 54     |
| Luxury SUV            | 4429  | 57     |
| Cargo Van             | 2663  | 18     |
| Pickup Truck          | 1691  | 69     |
| Roadster              | 5632  | 26     |
| Muscle Car            | 4793  | 38     |
| Off-road Vehicle      | 1343  | 31     |
| Camper Van            | 9124  | 74     |
| Compact Car           | 3652  | 82     |
| Motorcycle            | 8842  | 49     |
| Electric SUV          | 9176  | 64     |

#### Capacity

Total inventory capacity: $1576$