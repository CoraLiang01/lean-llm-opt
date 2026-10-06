Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the ProductName in the order given below. All $x_i$ are nonnegative integers.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value for product $i$.

Constraint (Total Inventory Capacity):
$$
\sum_{i} w_i x_i \leq 1576
$$
where $w_i$ is the Weight for product $i$.

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:

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

Complete Model:

$$
\begin{align*}
\max\ & 1752x_1 + 1856x_2 + 8372x_3 + 6168x_4 + 9681x_5 + 8062x_6 + 3895x_7 + 3254x_8 \\
     & + 1701x_9 + 6799x_{10} + 2724x_{11} + 6304x_{12} + 3255x_{13} + 1923x_{14} + 4103x_{15} \\
     & + 4429x_{16} + 2663x_{17} + 1691x_{18} + 5632x_{19} + 4793x_{20} + 1343x_{21} + 9124x_{22} \\
     & + 3652x_{23} + 8842x_{24} + 9176x_{25} \\
\text{s.t. } & 15x_1 + 87x_2 + 36x_3 + 30x_4 + 33x_5 + 72x_6 + 75x_7 + 71x_8 + 51x_9 + 21x_{10} \\
     & + 97x_{11} + 52x_{12} + 25x_{13} + 15x_{14} + 54x_{15} + 57x_{16} + 18x_{17} + 69x_{18} \\
     & + 26x_{19} + 38x_{20} + 31x_{21} + 74x_{22} + 82x_{23} + 49x_{24} + 64x_{25} \leq 1576 \\
     & x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,25
\end{align*}
$$