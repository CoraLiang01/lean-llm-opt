Let the set of vehicle types (in the order given) be indexed by i, with ProductName as below:

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
x_i = number of vehicles of type i to order per day, for i = 1,...,25.

Domain:
x_i ∈ {0, 1, 2, ...} (nonnegative integers), for all i.

Parameters (from products.csv, in order):

| i  | ProductName         | Value (profit) | Weight (stock units) |
|----|---------------------|---------------|----------------------|
| 1  | Sedan               | 2524          | 99                   |
| 2  | SUV                 | 4614          | 55                   |
| 3  | Truck               | 8416          | 75                   |
| 4  | Convertible         | 5917          | 94                   |
| 5  | Minivan             | 9048          | 80                   |
| 6  | Coupe               | 1140          | 82                   |
| 7  | Hatchback           | 8962          | 71                   |
| 8  | Station Wagon       | 1888          | 100                  |
| 9  | Electric Car        | 8487          | 28                   |
| 10 | Hybrid Car          | 4425          | 93                   |
| 11 | Luxury Sedan        | 4717          | 84                   |
| 12 | Sports Car          | 4210          | 83                   |
| 13 | Crossover           | 1226          | 62                   |
| 14 | Diesel Truck        | 7400          | 90                   |
| 15 | Compact SUV         | 4639          | 99                   |
| 16 | Luxury SUV          | 7712          | 96                   |
| 17 | Cargo Van           | 3299          | 21                   |
| 18 | Pickup Truck        | 9895          | 39                   |
| 19 | Roadster            | 4496          | 99                   |
| 20 | Muscle Car          | 4526          | 81                   |
| 21 | Off-road Vehicle    | 5688          | 6                    |
| 22 | Camper Van          | 3007          | 58                   |
| 23 | Compact Car         | 3623          | 37                   |
| 24 | Motorcycle          | 8474          | 15                   |
| 25 | Electric SUV        | 8372          | 37                   |

Total inventory capacity (from capacity.csv): 765

Mathematical Model:

Decision variables:
 x_i ∈ {0, 1, 2, ...} for i = 1,...,25

Objective:
 Maximize total profit:
  Maximize 2524 x₁ + 4614 x₂ + 8416 x₃ + 5917 x₄ + 9048 x₅ + 1140 x₆ + 8962 x₇ + 1888 x₈ + 8487 x₉ + 4425 x₁₀ + 4717 x₁₁ + 4210 x₁₂ + 1226 x₁₃ + 7400 x₁₄ + 4639 x₁₅ + 7712 x₁₆ + 3299 x₁₇ + 9895 x₁₈ + 4496 x₁₉ + 4526 x₂₀ + 5688 x₂₁ + 3007 x₂₂ + 3623 x₂₃ + 8474 x₂₄ + 8372 x₂₅

Subject to:

Inventory capacity constraint:
 99 x₁ + 55 x₂ + 75 x₃ + 94 x₄ + 80 x₅ + 82 x₆ + 71 x₇ + 100 x₈ + 28 x₉ + 93 x₁₀ + 84 x₁₁ + 83 x₁₂ + 62 x₁₃ + 90 x₁₄ + 99 x₁₅ + 96 x₁₆ + 21 x₁₇ + 39 x₁₈ + 99 x₁₉ + 81 x₂₀ + 6 x₂₁ + 58 x₂₂ + 37 x₂₃ + 15 x₂₄ + 37 x₂₅ ≤ 765

x_i ∈ {0, 1, 2, ...} for all i = 1,...,25

All coefficients and constraints are taken directly from the supplied data, preserving order and identifiers.