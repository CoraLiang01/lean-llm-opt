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

Let $p_i$ be the Value (profit) for each vehicle type, and $w_i$ be the Weight (stock space required) for each vehicle type, as given below:

| $i$ | ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|----------------|
| 1   | Sedan               | 2524          | 99             |
| 2   | SUV                 | 4614          | 55             |
| 3   | Truck               | 8416          | 75             |
| 4   | Convertible         | 5917          | 94             |
| 5   | Minivan             | 9048          | 80             |
| 6   | Coupe               | 1140          | 82             |
| 7   | Hatchback           | 8962          | 71             |
| 8   | Station Wagon       | 1888          | 100            |
| 9   | Electric Car        | 8487          | 28             |
| 10  | Hybrid Car          | 4425          | 93             |
| 11  | Luxury Sedan        | 4717          | 84             |
| 12  | Sports Car          | 4210          | 83             |
| 13  | Crossover           | 1226          | 62             |
| 14  | Diesel Truck        | 7400          | 90             |
| 15  | Compact SUV         | 4639          | 99             |
| 16  | Luxury SUV          | 7712          | 96             |
| 17  | Cargo Van           | 3299          | 21             |
| 18  | Pickup Truck        | 9895          | 39             |
| 19  | Roadster            | 4496          | 99             |
| 20  | Muscle Car          | 4526          | 81             |
| 21  | Off-road Vehicle    | 5688          | 6              |
| 22  | Camper Van          | 3007          | 58             |
| 23  | Compact Car         | 3623          | 37             |
| 24  | Motorcycle          | 8474          | 15             |
| 25  | Electric SUV        | 8372          | 37             |

The overall inventory capacity is $C = 765$.

The complete mathematical model is:

Objective:
$$
\max \sum_{i=1}^{25} p_i x_i
$$

Subject to:
$$
\sum_{i=1}^{25} w_i x_i \leq 765
$$

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

Where:
- $p_i$ and $w_i$ are as listed above for each ProductName,
- $x_i$ is the number of vehicles of type $i$ to order per day (nonnegative integer),
- The total stock space used does not exceed the overall capacity of 765.