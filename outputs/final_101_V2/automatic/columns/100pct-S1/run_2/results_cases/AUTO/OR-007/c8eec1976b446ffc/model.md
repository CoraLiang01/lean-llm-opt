Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the ProductName column of products.csv.

Objective:
\[
\max \sum_{i} p_i x_i
\]
where $p_i$ is the Value (profit) for vehicle type $i$.

Subject to:

Capacity constraint:
\[
\sum_{i} w_i x_i \leq 765
\]
where $w_i$ is the Weight for vehicle type $i$, and 765 is the overall inventory capacity from capacity.csv.

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where the data for each vehicle type $i$ is:

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Hybrid Car          | 4425  | 93     |
| Sports Car          | 4210  | 83     |
| Camper Van          | 3007  | 58     |
| SUV                 | 4614  | 55     |
| Muscle Car          | 4526  | 81     |
| Electric SUV        | 8372  | 37     |
| Minivan             | 9048  | 80     |
| Cargo Van           | 3299  | 21     |
| Compact Car         | 3623  | 37     |
| Hatchback           | 8962  | 71     |
| Off-road Vehicle    | 5688  | 6      |
| Coupe               | 1140  | 82     |
| Compact SUV         | 4639  | 99     |
| Sedan               | 2524  | 99     |
| Station Wagon       | 1888  | 100    |
| Diesel Truck        | 7400  | 90     |
| Roadster            | 4496  | 99     |
| Convertible         | 5917  | 94     |
| Electric Car        | 8487  | 28     |
| Luxury Sedan        | 4717  | 84     |
| Crossover           | 1226  | 62     |
| Pickup Truck        | 9895  | 39     |
| Motorcycle          | 8474  | 15     |
| Truck               | 8416  | 75     |
| Luxury SUV          | 7712  | 96     |

Decision variables:
\[
x_i = \text{number of units of vehicle type } i \text{ to order per day}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]

Complete model:
\[
\max \left(
4425x_{\text{Hybrid Car}} +
4210x_{\text{Sports Car}} +
3007x_{\text{Camper Van}} +
4614x_{\text{SUV}} +
4526x_{\text{Muscle Car}} +
8372x_{\text{Electric SUV}} +
9048x_{\text{Minivan}} +
3299x_{\text{Cargo Van}} +
3623x_{\text{Compact Car}} +
8962x_{\text{Hatchback}} +
5688x_{\text{Off-road Vehicle}} +
1140x_{\text{Coupe}} +
4639x_{\text{Compact SUV}} +
2524x_{\text{Sedan}} +
1888x_{\text{Station Wagon}} +
7400x_{\text{Diesel Truck}} +
4496x_{\text{Roadster}} +
5917x_{\text{Convertible}} +
8487x_{\text{Electric Car}} +
4717x_{\text{Luxury Sedan}} +
1226x_{\text{Crossover}} +
9895x_{\text{Pickup Truck}} +
8474x_{\text{Motorcycle}} +
8416x_{\text{Truck}} +
7712x_{\text{Luxury SUV}}
\right)
\]
subject to
\[
93x_{\text{Hybrid Car}} +
83x_{\text{Sports Car}} +
58x_{\text{Camper Van}} +
55x_{\text{SUV}} +
81x_{\text{Muscle Car}} +
37x_{\text{Electric SUV}} +
80x_{\text{Minivan}} +
21x_{\text{Cargo Van}} +
37x_{\text{Compact Car}} +
71x_{\text{Hatchback}} +
6x_{\text{Off-road Vehicle}} +
82x_{\text{Coupe}} +
99x_{\text{Compact SUV}} +
99x_{\text{Sedan}} +
100x_{\text{Station Wagon}} +
90x_{\text{Diesel Truck}} +
99x_{\text{Roadster}} +
94x_{\text{Convertible}} +
28x_{\text{Electric Car}} +
84x_{\text{Luxury Sedan}} +
62x_{\text{Crossover}} +
39x_{\text{Pickup Truck}} +
15x_{\text{Motorcycle}} +
75x_{\text{Truck}} +
96x_{\text{Luxury SUV}}
\leq 765
\]
and
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]