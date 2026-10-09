Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order retrieved.

#### Sets and Parameters

- Let $I$ be the set of vehicle types (ProductName), as listed below.
- For each $i \in I$:
    - $v_i$ = Value (benefit coefficient) of vehicle type $i$
    - $w_i$ = Weight (inventory space required per unit) of vehicle type $i$
- Let $C$ = 1576 (total inventory capacity, from capacity.csv)

#### Vehicle Types and Parameters (in source order):

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

#### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

#### Objective

$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraint

$$
\sum_{i \in I} w_i x_i \leq 1576
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Explicit Model (with all coefficients):

Let the vehicle types be indexed in the order above, i.e., $x_1$ = Sedan, $x_2$ = SUV, ..., $x_{25}$ = Electric SUV.

**Objective:**
\[
\max \Big(
1752 x_1 + 1856 x_2 + 8372 x_3 + 6168 x_4 + 9681 x_5 + 8062 x_6 + 3895 x_7 + 3254 x_8 + 1701 x_9 + 6799 x_{10} + 2724 x_{11} + 6304 x_{12} + 3255 x_{13} + 1923 x_{14} + 4103 x_{15} + 4429 x_{16} + 2663 x_{17} + 1691 x_{18} + 5632 x_{19} + 4793 x_{20} + 1343 x_{21} + 9124 x_{22} + 3652 x_{23} + 8842 x_{24} + 9176 x_{25}
\Big)
\]

**Subject to:**
\[
15 x_1 + 87 x_2 + 36 x_3 + 30 x_4 + 33 x_5 + 72 x_6 + 75 x_7 + 71 x_8 + 51 x_9 + 21 x_{10} + 97 x_{11} + 52 x_{12} + 25 x_{13} + 15 x_{14} + 54 x_{15} + 57 x_{16} + 18 x_{17} + 69 x_{18} + 26 x_{19} + 38 x_{20} + 31 x_{21} + 74 x_{22} + 82 x_{23} + 49 x_{24} + 64 x_{25} \leq 1576
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

Where the mapping of $x_i$ to ProductName is as listed above.