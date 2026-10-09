Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following vehicle types in the order given:

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

Let $v_i$ be the benefit (Value) and $w_i$ be the weight (Weight) for each vehicle type $i$, as given below.

Let $C$ be the total inventory capacity.

#### Parameters (from data):

| $i$ | ProductName         | $v_i$ (Value) | $w_i$ (Weight) |
|-----|---------------------|---------------|---------------|
| 1   | Sedan               | 1752          | 15            |
| 2   | SUV                 | 1856          | 87            |
| 3   | Truck               | 8372          | 36            |
| 4   | Convertible         | 6168          | 30            |
| 5   | Minivan             | 9681          | 33            |
| 6   | Coupe               | 8062          | 72            |
| 7   | Hatchback           | 3895          | 75            |
| 8   | Station Wagon       | 3254          | 71            |
| 9   | Electric Car        | 1701          | 51            |
| 10  | Hybrid Car          | 6799          | 21            |
| 11  | Luxury Sedan        | 2724          | 97            |
| 12  | Sports Car          | 6304          | 52            |
| 13  | Crossover           | 3255          | 25            |
| 14  | Diesel Truck        | 1923          | 15            |
| 15  | Compact SUV         | 4103          | 54            |
| 16  | Luxury SUV          | 4429          | 57            |
| 17  | Cargo Van           | 2663          | 18            |
| 18  | Pickup Truck        | 1691          | 69            |
| 19  | Roadster            | 5632          | 26            |
| 20  | Muscle Car          | 4793          | 38            |
| 21  | Off-road Vehicle    | 1343          | 31            |
| 22  | Camper Van          | 9124          | 74            |
| 23  | Compact Car         | 3652          | 82            |
| 24  | Motorcycle          | 8842          | 49            |
| 25  | Electric SUV        | 9176          | 64            |

Total inventory capacity: $C = 1576$

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
\sum_{i=1}^{25} w_i x_i \leq 1576
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
$$

Where the $v_i$ and $w_i$ are as listed above for each vehicle type $i$.