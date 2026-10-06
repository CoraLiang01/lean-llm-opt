Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the following ProductName values from products.csv.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value for product $i$.

Subject to:

Inventory capacity constraint:
$$
\sum_{i} w_i x_i \leq 1576
$$
where $w_i$ is the Weight for product $i$.

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Sedan               | 1752  | 15     |
| SUV                 | 1856  | 87     |
| Truck               | 8372  | 36     |
| Convertible         | 6168  | 30     |
| Minivan             | 9681  | 33     |
| Coupe               | 8062  | 72     |
| Hatchback           | 3895  | 75     |
| Station Wagon       | 3254  | 71     |
| Electric Car        | 1701  | 51     |
| Hybrid Car          | 6799  | 21     |
| Luxury Sedan        | 2724  | 97     |
| Sports Car          | 6304  | 52     |
| Crossover           | 3255  | 25     |
| Diesel Truck        | 1923  | 15     |
| Compact SUV         | 4103  | 54     |
| Luxury SUV          | 4429  | 57     |
| Cargo Van           | 2663  | 18     |
| Pickup Truck        | 1691  | 69     |
| Roadster            | 5632  | 26     |
| Muscle Car          | 4793  | 38     |
| Off-road Vehicle    | 1343  | 31     |
| Camper Van          | 9124  | 74     |
| Compact Car         | 3652  | 82     |
| Motorcycle          | 8842  | 49     |
| Electric SUV        | 9176  | 64     |

Decision variables:
$$
x_i = \text{number of units of vehicle type } i \text{ to order daily}, \quad x_i \in \mathbb{Z}_{\geq 0}
$$

Complete Model:
$$
\begin{align*}
\max \quad & 1752x_{\text{Sedan}} + 1856x_{\text{SUV}} + 8372x_{\text{Truck}} + 6168x_{\text{Convertible}} + 9681x_{\text{Minivan}} + 8062x_{\text{Coupe}} + 3895x_{\text{Hatchback}} \\
& + 3254x_{\text{Station Wagon}} + 1701x_{\text{Electric Car}} + 6799x_{\text{Hybrid Car}} + 2724x_{\text{Luxury Sedan}} + 6304x_{\text{Sports Car}} + 3255x_{\text{Crossover}} \\
& + 1923x_{\text{Diesel Truck}} + 4103x_{\text{Compact SUV}} + 4429x_{\text{Luxury SUV}} + 2663x_{\text{Cargo Van}} + 1691x_{\text{Pickup Truck}} + 5632x_{\text{Roadster}} \\
& + 4793x_{\text{Muscle Car}} + 1343x_{\text{Off-road Vehicle}} + 9124x_{\text{Camper Van}} + 3652x_{\text{Compact Car}} + 8842x_{\text{Motorcycle}} + 9176x_{\text{Electric SUV}} \\
\text{s.t.} \quad & 15x_{\text{Sedan}} + 87x_{\text{SUV}} + 36x_{\text{Truck}} + 30x_{\text{Convertible}} + 33x_{\text{Minivan}} + 72x_{\text{Coupe}} + 75x_{\text{Hatchback}} \\
& + 71x_{\text{Station Wagon}} + 51x_{\text{Electric Car}} + 21x_{\text{Hybrid Car}} + 97x_{\text{Luxury Sedan}} + 52x_{\text{Sports Car}} + 25x_{\text{Crossover}} \\
& + 15x_{\text{Diesel Truck}} + 54x_{\text{Compact SUV}} + 57x_{\text{Luxury SUV}} + 18x_{\text{Cargo Van}} + 69x_{\text{Pickup Truck}} + 26x_{\text{Roadster}} \\
& + 38x_{\text{Muscle Car}} + 31x_{\text{Off-road Vehicle}} + 74x_{\text{Camper Van}} + 82x_{\text{Compact Car}} + 49x_{\text{Motorcycle}} + 64x_{\text{Electric SUV}} \leq 1576 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$