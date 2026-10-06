Let the set of vehicle types be indexed by i, corresponding to the rows in products.csv, with ProductName_i, Value_i, and Weight_i as given below. Let x_i be the integer number of units of vehicle type i to order daily.

Data (in original file order):

| i | ProductName        | Value_i | Weight_i |
|---|--------------------|---------|----------|
| 1 | Sedan              | 1752    | 15       |
| 2 | SUV                | 1856    | 87       |
| 3 | Truck              | 8372    | 36       |
| 4 | Convertible        | 6168    | 30       |
| 5 | Minivan            | 9681    | 33       |
| 6 | Coupe              | 8062    | 72       |
| 7 | Hatchback          | 3895    | 75       |
| 8 | Station Wagon      | 3254    | 71       |
| 9 | Electric Car       | 1701    | 51       |
|10 | Hybrid Car         | 6799    | 21       |
|11 | Luxury Sedan       | 2724    | 97       |
|12 | Sports Car         | 6304    | 52       |
|13 | Crossover          | 3255    | 25       |
|14 | Diesel Truck       | 1923    | 15       |
|15 | Compact SUV        | 4103    | 54       |
|16 | Luxury SUV         | 4429    | 57       |
|17 | Cargo Van          | 2663    | 18       |
|18 | Pickup Truck       | 1691    | 69       |
|19 | Roadster           | 5632    | 26       |
|20 | Muscle Car         | 4793    | 38       |
|21 | Off-road Vehicle   | 1343    | 31       |
|22 | Camper Van         | 9124    | 74       |
|23 | Compact Car        | 3652    | 82       |
|24 | Motorcycle         | 8842    | 49       |
|25 | Electric SUV       | 9176    | 64       |

Total inventory capacity: 1576

Mathematical Model:

Decision variables:
For i = 1,...,25,
 x_i = number of units of vehicle type i to order daily (x_i ∈ {0, 1, 2, ...})

Objective:
Maximize total benefit:
 maximize Z = 1752 x_1 + 1856 x_2 + 8372 x_3 + 6168 x_4 + 9681 x_5 + 8062 x_6 + 3895 x_7 + 3254 x_8 + 1701 x_9 + 6799 x_10 + 2724 x_11 + 6304 x_12 + 3255 x_13 + 1923 x_14 + 4103 x_15 + 4429 x_16 + 2663 x_17 + 1691 x_18 + 5632 x_19 + 4793 x_20 + 1343 x_21 + 9124 x_22 + 3652 x_23 + 8842 x_24 + 9176 x_25

Subject to:

Inventory capacity constraint:
 15 x_1 + 87 x_2 + 36 x_3 + 30 x_4 + 33 x_5 + 72 x_6 + 75 x_7 + 71 x_8 + 51 x_9 + 21 x_10 + 97 x_11 + 52 x_12 + 25 x_13 + 15 x_14 + 54 x_15 + 57 x_16 + 18 x_17 + 69 x_18 + 26 x_19 + 38 x_20 + 31 x_21 + 74 x_22 + 82 x_23 + 49 x_24 + 64 x_25 ≤ 1576

Integrality:
 x_i ∈ {0, 1, 2, ...} for all i = 1,...,25

All coefficients and constraints are taken directly from the supplied CSV files.