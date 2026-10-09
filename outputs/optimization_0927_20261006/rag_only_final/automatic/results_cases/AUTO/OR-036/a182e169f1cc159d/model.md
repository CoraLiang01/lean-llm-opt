Let the set of vehicle types be indexed by i, corresponding to the following ProductName values in the order given:

1. Sedan
2. SUV
3. Truck
4. Convertible
5. Minivan
6. Coupe
7. Hatchback
8. Station Wagon
9. Electric Car
10. Hybrid Car
11. Luxury Sedan
12. Sports Car
13. Crossover
14. Diesel Truck
15. Compact SUV
16. Luxury SUV
17. Cargo Van
18. Pickup Truck
19. Roadster
20. Muscle Car
21. Off-road Vehicle
22. Camper Van
23. Compact Car
24. Motorcycle
25. Electric SUV

Define decision variables:
x_i = number of units of vehicle type i to order daily, for i = 1,...,25

All x_i are nonnegative integers.

Parameters (from products.csv, in order):

| i  | ProductName        | Value | Weight |
|----|--------------------|-------|--------|
| 1  | Sedan              | 1752  | 15     |
| 2  | SUV                | 1856  | 87     |
| 3  | Truck              | 8372  | 36     |
| 4  | Convertible        | 6168  | 30     |
| 5  | Minivan            | 9681  | 33     |
| 6  | Coupe              | 8062  | 72     |
| 7  | Hatchback          | 3895  | 75     |
| 8  | Station Wagon      | 3254  | 71     |
| 9  | Electric Car       | 1701  | 51     |
| 10 | Hybrid Car         | 6799  | 21     |
| 11 | Luxury Sedan       | 2724  | 97     |
| 12 | Sports Car         | 6304  | 52     |
| 13 | Crossover          | 3255  | 25     |
| 14 | Diesel Truck       | 1923  | 15     |
| 15 | Compact SUV        | 4103  | 54     |
| 16 | Luxury SUV         | 4429  | 57     |
| 17 | Cargo Van          | 2663  | 18     |
| 18 | Pickup Truck       | 1691  | 69     |
| 19 | Roadster           | 5632  | 26     |
| 20 | Muscle Car         | 4793  | 38     |
| 21 | Off-road Vehicle   | 1343  | 31     |
| 22 | Camper Van         | 9124  | 74     |
| 23 | Compact Car        | 3652  | 82     |
| 24 | Motorcycle         | 8842  | 49     |
| 25 | Electric SUV       | 9176  | 64     |

Total inventory capacity (from capacity.csv): 1576

Mathematical Model:

Variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,25

Objective:
 Maximize total benefit:
  maximize 1752 x₁ + 1856 x₂ + 8372 x₃ + 6168 x₄ + 9681 x₅ + 8062 x₆ + 3895 x₇ + 3254 x₈ + 1701 x₉ + 6799 x₁₀ + 2724 x₁₁ + 6304 x₁₂ + 3255 x₁₃ + 1923 x₁₄ + 4103 x₁₅ + 4429 x₁₆ + 2663 x₁₇ + 1691 x₁₈ + 5632 x₁₉ + 4793 x₂₀ + 1343 x₂₁ + 9124 x₂₂ + 3652 x₂₃ + 8842 x₂₄ + 9176 x₂₅

Subject to:
 Inventory capacity constraint:
  15 x₁ + 87 x₂ + 36 x₃ + 30 x₄ + 33 x₅ + 72 x₆ + 75 x₇ + 71 x₈ + 51 x₉ + 21 x₁₀ + 97 x₁₁ + 52 x₁₂ + 25 x₁₃ + 15 x₁₄ + 54 x₁₅ + 57 x₁₆ + 18 x₁₇ + 69 x₁₈ + 26 x₁₉ + 38 x₂₀ + 31 x₂₁ + 74 x₂₂ + 82 x₂₃ + 49 x₂₄ + 64 x₂₅ ≤ 1576

 x_i ≥ 0 and integer for all i = 1,...,25

Where x₁ = Sedan, x₂ = SUV, ..., x₂₅ = Electric SUV, as listed above.