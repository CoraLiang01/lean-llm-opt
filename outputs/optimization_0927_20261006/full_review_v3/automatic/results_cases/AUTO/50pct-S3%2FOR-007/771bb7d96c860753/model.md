Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order given:

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

Profits $p_i$ and weights $w_i$ for each vehicle type are as follows (in source order):

| ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------|--------------|---------------|
| Sedan               | 2524         | 99            |
| SUV                 | 4614         | 55            |
| Truck               | 8416         | 75            |
| Convertible         | 5917         | 94            |
| Minivan             | 9048         | 80            |
| Coupe               | 1140         | 82            |
| Hatchback           | 8962         | 71            |
| Station Wagon       | 1888         | 100           |
| Electric Car        | 8487         | 28            |
| Hybrid Car          | 4425         | 93            |
| Luxury Sedan        | 4717         | 84            |
| Sports Car          | 4210         | 83            |
| Crossover           | 1226         | 62            |
| Diesel Truck        | 7400         | 90            |
| Compact SUV         | 4639         | 99            |
| Luxury SUV          | 7712         | 96            |
| Cargo Van           | 3299         | 21            |
| Pickup Truck        | 9895         | 39            |
| Roadster            | 4496         | 99            |
| Muscle Car          | 4526         | 81            |
| Off-road Vehicle    | 5688         | 6             |
| Camper Van          | 3007         | 58            |
| Compact Car         | 3623         | 37            |
| Motorcycle          | 8474         | 15            |
| Electric SUV        | 8372         | 37            |

The overall inventory capacity is:

- Capacity = 765

##### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{25} p_i x_i
\]

**Subject to:**

\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = profit (Value) for vehicle type $i$ (see table above)
- $w_i$ = weight (Weight) for vehicle type $i$ (see table above)
- $765$ = total inventory capacity

**All data and identifiers are preserved in source order.**