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

Let $v_i$ be the Value (profit) and $w_i$ be the Weight (stock space required) for each product $i$, as given below:

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedan               | 2524  | 99     |
| SUV                 | 4614  | 55     |
| Truck               | 8416  | 75     |
| Convertible         | 5917  | 94     |
| Minivan             | 9048  | 80     |
| Coupe               | 1140  | 82     |
| Hatchback           | 8962  | 71     |
| Station Wagon       | 1888  | 100    |
| Electric Car        | 8487  | 28     |
| Hybrid Car          | 4425  | 93     |
| Luxury Sedan        | 4717  | 84     |
| Sports Car          | 4210  | 83     |
| Crossover           | 1226  | 62     |
| Diesel Truck        | 7400  | 90     |
| Compact SUV         | 4639  | 99     |
| Luxury SUV          | 7712  | 96     |
| Cargo Van           | 3299  | 21     |
| Pickup Truck        | 9895  | 39     |
| Roadster            | 4496  | 99     |
| Muscle Car          | 4526  | 81     |
| Off-road Vehicle    | 5688  | 6      |
| Camper Van          | 3007  | 58     |
| Compact Car         | 3623  | 37     |
| Motorcycle          | 8474  | 15     |
| Electric SUV        | 8372  | 37     |

The total inventory capacity is $765$ (from the "Capacity" column in capacity.csv).

The mathematical model is:

$$
\begin{align*}
\max \quad & \sum_{i=1}^{25} v_i x_i \\
\text{s.t.} \quad & \sum_{i=1}^{25} w_i x_i \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
\end{align*}
$$

Where:
- $v_i$ and $w_i$ are as listed above for each ProductName.
- $x_i$ is the number of vehicles of type $i$ to order per day (nonnegative integer).

All data and identifiers are preserved in original order.