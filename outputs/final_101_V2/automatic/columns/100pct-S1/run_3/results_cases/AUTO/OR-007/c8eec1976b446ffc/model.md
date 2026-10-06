Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the ProductName in the order given below. Each $x_i$ is a nonnegative integer.

**Objective:**
\[
\max \sum_{i} p_i x_i
\]
where $p_i$ is the Value (profit) for vehicle type $i$.

**Constraint:**
\[
\sum_{i} w_i x_i \leq 765
\]
where $w_i$ is the Weight for vehicle type $i$.

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Data (in source order):**

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

**Capacity:**
\[
765
\]

**Complete Model:**

\[
\max \Big(
2524\,x_1 + 4614\,x_2 + 8416\,x_3 + 5917\,x_4 + 9048\,x_5 + 1140\,x_6 + 8962\,x_7 + 1888\,x_8 + 8487\,x_9 + 4425\,x_{10} + 4210\,x_{11} + 1226\,x_{12} + 7400\,x_{13} + 4639\,x_{14} + 7712\,x_{15} + 3299\,x_{16} + 9895\,x_{17} + 4496\,x_{18} + 4526\,x_{19} + 5688\,x_{20} + 3007\,x_{21} + 3623\,x_{22} + 8474\,x_{23} + 8372\,x_{24}
\Big)
\]

subject to

\[
99\,x_1 + 55\,x_2 + 75\,x_3 + 94\,x_4 + 80\,x_5 + 82\,x_6 + 71\,x_7 + 100\,x_8 + 28\,x_9 + 93\,x_{10} + 83\,x_{11} + 62\,x_{12} + 90\,x_{13} + 99\,x_{14} + 96\,x_{15} + 21\,x_{16} + 39\,x_{17} + 99\,x_{18} + 81\,x_{19} + 6\,x_{20} + 58\,x_{21} + 37\,x_{22} + 15\,x_{23} + 37\,x_{24} \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 24
\]

where the mapping of $i$ to ProductName is as listed above.