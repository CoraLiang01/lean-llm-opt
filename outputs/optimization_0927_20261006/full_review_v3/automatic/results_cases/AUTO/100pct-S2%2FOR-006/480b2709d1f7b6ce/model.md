Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order retrieved:

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

The benefit coefficients and weights for each vehicle type are as follows (in source order):

| $i$ | ProductName         | Value | Weight |
|-----|---------------------|-------|--------|
| 1   | Sedan               | 1752  | 15     |
| 2   | SUV                 | 1856  | 87     |
| 3   | Truck               | 8372  | 36     |
| 4   | Convertible         | 6168  | 30     |
| 5   | Minivan             | 9681  | 33     |
| 6   | Coupe               | 8062  | 72     |
| 7   | Hatchback           | 3895  | 75     |
| 8   | Station Wagon       | 3254  | 71     |
| 9   | Electric Car        | 1701  | 51     |
| 10  | Hybrid Car          | 6799  | 21     |
| 11  | Luxury Sedan        | 2724  | 97     |
| 12  | Sports Car          | 6304  | 52     |
| 13  | Crossover           | 3255  | 25     |
| 14  | Diesel Truck        | 1923  | 15     |
| 15  | Compact SUV         | 4103  | 54     |
| 16  | Luxury SUV          | 4429  | 57     |
| 17  | Cargo Van           | 2663  | 18     |
| 18  | Pickup Truck        | 1691  | 69     |
| 19  | Roadster            | 5632  | 26     |
| 20  | Muscle Car          | 4793  | 38     |
| 21  | Off-road Vehicle    | 1343  | 31     |
| 22  | Camper Van          | 9124  | 74     |
| 23  | Compact Car         | 3652  | 82     |
| 24  | Motorcycle          | 8842  | 49     |
| 25  | Electric SUV        | 9176  | 64     |

The total inventory capacity is 1576 units (from the "Capacity" column in capacity.csv).

The mathematical model is:

---

**Objective:**

$$
\max \sum_{i=1}^{25} v_i x_i
$$

where $v_i$ is the Value for vehicle type $i$ as listed above.

**Constraint:**

$$
\sum_{i=1}^{25} w_i x_i \leq 1576
$$

where $w_i$ is the Weight for vehicle type $i$ as listed above.

**Variable domains:**

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

---

**Explicitly:**

$$
\max \Big(
1752 x_1 + 1856 x_2 + 8372 x_3 + 6168 x_4 + 9681 x_5 + 8062 x_6 + 3895 x_7 + 3254 x_8 + 1701 x_9 + 6799 x_{10} + 2724 x_{11} + 6304 x_{12} + 3255 x_{13} + 1923 x_{14} + 4103 x_{15} + 4429 x_{16} + 2663 x_{17} + 1691 x_{18} + 5632 x_{19} + 4793 x_{20} + 1343 x_{21} + 9124 x_{22} + 3652 x_{23} + 8842 x_{24} + 9176 x_{25}
\Big)
$$

subject to

$$
15 x_1 + 87 x_2 + 36 x_3 + 30 x_4 + 33 x_5 + 72 x_6 + 75 x_7 + 71 x_8 + 51 x_9 + 21 x_{10} + 97 x_{11} + 52 x_{12} + 25 x_{13} + 15 x_{14} + 54 x_{15} + 57 x_{16} + 18 x_{17} + 69 x_{18} + 26 x_{19} + 38 x_{20} + 31 x_{21} + 74 x_{22} + 82 x_{23} + 49 x_{24} + 64 x_{25} \leq 1576
$$

and

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

---

All coefficients and identifiers are as retrieved and in source order.