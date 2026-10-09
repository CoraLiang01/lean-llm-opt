Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values in the order given:

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

Parameters:

- $v_i$: Value (benefit) of vehicle type $i$
- $w_i$: Weight (inventory space required) of vehicle type $i$
- $C$: Total inventory capacity

Data:

| ProductName         | $v_i$ | $w_i$ |
|---------------------|-------|-------|
| Sedan               | 1752  | 15    |
| SUV                 | 1856  | 87    |
| Truck               | 8372  | 36    |
| Convertible         | 6168  | 30    |
| Minivan             | 9681  | 33    |
| Coupe               | 8062  | 72    |
| Hatchback           | 3895  | 75    |
| Station Wagon       | 3254  | 71    |
| Electric Car        | 1701  | 51    |
| Hybrid Car          | 6799  | 21    |
| Luxury Sedan        | 2724  | 97    |
| Sports Car          | 6304  | 52    |
| Crossover           | 3255  | 25    |
| Diesel Truck        | 1923  | 15    |
| Compact SUV         | 4103  | 54    |
| Luxury SUV          | 4429  | 57    |
| Cargo Van           | 2663  | 18    |
| Pickup Truck        | 1691  | 69    |
| Roadster            | 5632  | 26    |
| Muscle Car          | 4793  | 38    |
| Off-road Vehicle    | 1343  | 31    |
| Camper Van          | 9124  | 74    |
| Compact Car         | 3652  | 82    |
| Motorcycle          | 8842  | 49    |
| Electric SUV        | 9176  | 64    |

Total inventory capacity: $C = 1576$

Model:

Maximize total benefit:
$$
\max \sum_{i=1}^{25} v_i x_i
$$

Subject to the total inventory capacity:
$$
\sum_{i=1}^{25} w_i x_i \leq 1576
$$

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

Where the parameters $v_i$ and $w_i$ are as listed above for each ProductName.