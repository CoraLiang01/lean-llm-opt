Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order retrieved:

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

The profit (Value) and weight (Weight) for each vehicle type are as follows (in the same order):

| $i$ | ProductName         | Value | Weight |
|-----|---------------------|-------|--------|
| 1   | Sedan               | 2524  | 99     |
| 2   | SUV                 | 4614  | 55     |
| 3   | Truck               | 8416  | 75     |
| 4   | Convertible         | 5917  | 94     |
| 5   | Minivan             | 9048  | 80     |
| 6   | Coupe               | 1140  | 82     |
| 7   | Hatchback           | 8962  | 71     |
| 8   | Station Wagon       | 1888  | 100    |
| 9   | Electric Car        | 8487  | 28     |
| 10  | Hybrid Car          | 4425  | 93     |
| 11  | Luxury Sedan        | 4717  | 84     |
| 12  | Sports Car          | 4210  | 83     |
| 13  | Crossover           | 1226  | 62     |
| 14  | Diesel Truck        | 7400  | 90     |
| 15  | Compact SUV         | 4639  | 99     |
| 16  | Luxury SUV          | 7712  | 96     |
| 17  | Cargo Van           | 3299  | 21     |
| 18  | Pickup Truck        | 9895  | 39     |
| 19  | Roadster            | 4496  | 99     |
| 20  | Muscle Car          | 4526  | 81     |
| 21  | Off-road Vehicle    | 5688  | 6      |
| 22  | Camper Van          | 3007  | 58     |
| 23  | Compact Car         | 3623  | 37     |
| 24  | Motorcycle          | 8474  | 15     |
| 25  | Electric SUV        | 8372  | 37     |

The total inventory capacity is:

$\text{Capacity} = 765$

The mathematical model is:

**Objective:**
\[
\max \left(
2524x_1 + 4614x_2 + 8416x_3 + 5917x_4 + 9048x_5 + 1140x_6 + 8962x_7 + 1888x_8 + 8487x_9 + 4425x_{10} + 4717x_{11} + 4210x_{12} + 1226x_{13} + 7400x_{14} + 4639x_{15} + 7712x_{16} + 3299x_{17} + 9895x_{18} + 4496x_{19} + 4526x_{20} + 5688x_{21} + 3007x_{22} + 3623x_{23} + 8474x_{24} + 8372x_{25}
\right)
\]

**Subject to:**
\[
99x_1 + 55x_2 + 75x_3 + 94x_4 + 80x_5 + 82x_6 + 71x_7 + 100x_8 + 28x_9 + 93x_{10} + 84x_{11} + 83x_{12} + 62x_{13} + 90x_{14} + 99x_{15} + 96x_{16} + 21x_{17} + 39x_{18} + 99x_{19} + 81x_{20} + 6x_{21} + 58x_{22} + 37x_{23} + 15x_{24} + 37x_{25} \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

**Where:**
- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- Value and Weight as above
- Total inventory capacity is 765 units (as per the "Capacity" field in capacity.csv)