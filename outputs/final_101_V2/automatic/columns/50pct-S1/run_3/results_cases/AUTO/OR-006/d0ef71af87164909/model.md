Let $x_i$ be the number of units of vehicle type $i$ to order daily, where $i$ indexes the 25 vehicle types listed below. All $x_i$ are required to be nonnegative integers.

Objective:
\[
\max \left(
1752\,x_{\text{Sedan}} +
1856\,x_{\text{SUV}} +
8372\,x_{\text{Truck}} +
6168\,x_{\text{Convertible}} +
9681\,x_{\text{Minivan}} +
8062\,x_{\text{Coupe}} +
3895\,x_{\text{Hatchback}} +
3254\,x_{\text{Station Wagon}} +
1701\,x_{\text{Electric Car}} +
6799\,x_{\text{Hybrid Car}} +
2724\,x_{\text{Luxury Sedan}} +
6304\,x_{\text{Sports Car}} +
3255\,x_{\text{Crossover}} +
1923\,x_{\text{Diesel Truck}} +
4103\,x_{\text{Compact SUV}} +
4429\,x_{\text{Luxury SUV}} +
2663\,x_{\text{Cargo Van}} +
1691\,x_{\text{Pickup Truck}} +
5632\,x_{\text{Roadster}} +
4793\,x_{\text{Muscle Car}} +
1343\,x_{\text{Off-road Vehicle}} +
9124\,x_{\text{Camper Van}} +
3652\,x_{\text{Compact Car}} +
8842\,x_{\text{Motorcycle}} +
9176\,x_{\text{Electric SUV}}
\right)
\]

Subject to:

Inventory capacity constraint:
\[
15\,x_{\text{Sedan}} +
87\,x_{\text{SUV}} +
36\,x_{\text{Truck}} +
30\,x_{\text{Convertible}} +
33\,x_{\text{Minivan}} +
72\,x_{\text{Coupe}} +
75\,x_{\text{Hatchback}} +
71\,x_{\text{Station Wagon}} +
51\,x_{\text{Electric Car}} +
21\,x_{\text{Hybrid Car}} +
97\,x_{\text{Luxury Sedan}} +
52\,x_{\text{Sports Car}} +
25\,x_{\text{Crossover}} +
15\,x_{\text{Diesel Truck}} +
54\,x_{\text{Compact SUV}} +
57\,x_{\text{Luxury SUV}} +
18\,x_{\text{Cargo Van}} +
69\,x_{\text{Pickup Truck}} +
26\,x_{\text{Roadster}} +
38\,x_{\text{Muscle Car}} +
31\,x_{\text{Off-road Vehicle}} +
74\,x_{\text{Camper Van}} +
82\,x_{\text{Compact Car}} +
49\,x_{\text{Motorcycle}} +
64\,x_{\text{Electric SUV}}
\leq 1576
\]

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where the vehicle types $i$ and their corresponding coefficients are:

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

All variables $x_i$ are nonnegative integers.