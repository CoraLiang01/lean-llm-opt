Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values in the order returned:

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

Let $p_i$ be the profit (Value) for vehicle type $i$, and $w_i$ be the Weight (stock space requirement) for vehicle type $i$.

Let $C$ be the overall inventory capacity.

The data is:

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

The overall inventory capacity is:

$C = 765$

The mathematical model is:

Objective:
$$
\max \sum_{i=1}^{25} p_i x_i
$$

where $p_i$ is as above.

Subject to:

Inventory capacity constraint:
$$
\sum_{i=1}^{25} w_i x_i \leq 765
$$

where $w_i$ is as above.

Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

Explicitly, the model is:

$$
\max \Big(
2524\,x_1 + 4614\,x_2 + 8416\,x_3 + 5917\,x_4 + 9048\,x_5 + 1140\,x_6 + 8962\,x_7 + 1888\,x_8 + 8487\,x_9 + 4425\,x_{10} + 4717\,x_{11} + 4210\,x_{12} + 1226\,x_{13} + 7400\,x_{14} + 4639\,x_{15} + 7712\,x_{16} + 3299\,x_{17} + 9895\,x_{18} + 4496\,x_{19} + 4526\,x_{20} + 5688\,x_{21} + 3007\,x_{22} + 3623\,x_{23} + 8474\,x_{24} + 8372\,x_{25}
\Big)
$$

subject to

$$
99\,x_1 + 55\,x_2 + 75\,x_3 + 94\,x_4 + 80\,x_5 + 82\,x_6 + 71\,x_7 + 100\,x_8 + 28\,x_9 + 93\,x_{10} + 84\,x_{11} + 83\,x_{12} + 62\,x_{13} + 90\,x_{14} + 99\,x_{15} + 96\,x_{16} + 21\,x_{17} + 39\,x_{18} + 99\,x_{19} + 81\,x_{20} + 6\,x_{21} + 58\,x_{22} + 37\,x_{23} + 15\,x_{24} + 37\,x_{25} \leq 765
$$

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
$$

where $x_i$ is the number of vehicles of type $i$ to order per day.