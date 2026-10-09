Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following vehicle types in the order given:

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

Let $v_i$ be the profit (Value) and $w_i$ be the weight (Weight) for each vehicle type $i$, as given below.

The total inventory capacity is $765$.

**Parameters (from products.csv, in source order):**

| $i$ | ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|---------------|
| 1   | Sedan               | 2524          | 99            |
| 2   | SUV                 | 4614          | 55            |
| 3   | Truck               | 8416          | 75            |
| 4   | Convertible         | 5917          | 94            |
| 5   | Minivan             | 9048          | 80            |
| 6   | Coupe               | 1140          | 82            |
| 7   | Hatchback           | 8962          | 71            |
| 8   | Station Wagon       | 1888          | 100           |
| 9   | Electric Car        | 8487          | 28            |
| 10  | Hybrid Car          | 4425          | 93            |
| 11  | Luxury Sedan        | 4717          | 84            |
| 12  | Sports Car          | 4210          | 83            |
| 13  | Crossover           | 1226          | 62            |
| 14  | Diesel Truck        | 7400          | 90            |
| 15  | Compact SUV         | 4639          | 99            |
| 16  | Luxury SUV          | 7712          | 96            |
| 17  | Cargo Van           | 3299          | 21            |
| 18  | Pickup Truck        | 9895          | 39            |
| 19  | Roadster            | 4496          | 99            |
| 20  | Muscle Car          | 4526          | 81            |
| 21  | Off-road Vehicle    | 5688          | 6             |
| 22  | Camper Van          | 3007          | 58            |
| 23  | Compact Car         | 3623          | 37            |
| 24  | Motorcycle          | 8474          | 15            |
| 25  | Electric SUV        | 8372          | 37            |

**Capacity (from capacity.csv):**

Total inventory capacity: $765$

---

### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
$$

**Objective:**
$$
\max \sum_{i=1}^{25} v_i x_i
$$

**Subject to:**
$$
\sum_{i=1}^{25} w_i x_i \leq 765
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
$$

**Where:**

- $v_i$ and $w_i$ are as listed above for each vehicle type $i$.
- $x_i$ is the number of vehicles of type $i$ to order per day.

**All data is used in the order and with the identifiers as retrieved.**