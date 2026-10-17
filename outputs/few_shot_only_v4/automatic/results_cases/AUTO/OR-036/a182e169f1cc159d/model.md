Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the vehicle types listed in the ProductName column. All $x_i$ are integer and nonnegative.

**Parameters:**

- $v_i$: Value (benefit) of vehicle type $i$ (from Value column)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from Weight column)
- $C$: Total inventory capacity (from Capacity.csv, $C = 1576$)

**Vehicle Types and Parameters:**

| $i$ | ProductName         | $v_i$ | $w_i$ |
|-----|---------------------|-------|-------|
| 1   | Sedan               | 1752  | 15    |
| 2   | SUV                 | 1856  | 87    |
| 3   | Truck               | 8372  | 36    |
| 4   | Convertible         | 6168  | 30    |
| 5   | Minivan             | 9681  | 33    |
| 6   | Coupe               | 8062  | 72    |
| 7   | Hatchback           | 3895  | 75    |
| 8   | Station Wagon       | 3254  | 71    |
| 9   | Electric Car        | 1701  | 51    |
| 10  | Hybrid Car          | 6799  | 21    |
| 11  | Luxury Sedan        | 2724  | 97    |
| 12  | Sports Car          | 6304  | 52    |
| 13  | Crossover           | 3255  | 25    |
| 14  | Diesel Truck        | 1923  | 15    |
| 15  | Compact SUV         | 4103  | 54    |
| 16  | Luxury SUV          | 4429  | 57    |
| 17  | Cargo Van           | 2663  | 18    |
| 18  | Pickup Truck        | 1691  | 69    |
| 19  | Roadster            | 5632  | 26    |
| 20  | Muscle Car          | 4793  | 38    |
| 21  | Off-road Vehicle    | 1343  | 31    |
| 22  | Camper Van          | 9124  | 74    |
| 23  | Compact Car         | 3652  | 82    |
| 24  | Motorcycle          | 8842  | 49    |
| 25  | Electric SUV        | 9176  | 64    |

**Mathematical Model:**

Objective:
\[
\max \sum_{i=1}^{25} v_i x_i
\]

Subject to:
\[
\sum_{i=1}^{25} w_i x_i \leq 1576
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

Where:
- $v_i$ and $w_i$ are as listed above for each vehicle type $i$.
- $x_i$ is the integer number of units of vehicle type $i$ to order daily.