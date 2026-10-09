Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values:

- Sedan
- SUV
- Truck
- Convertible
- Minivan
- Coupe
- Hatchback
- Station Wagon
- Electric Car
- Hybrid Car
- Luxury Sedan
- Sports Car
- Crossover
- Diesel Truck
- Compact SUV
- Luxury SUV
- Cargo Van
- Pickup Truck
- Roadster
- Muscle Car
- Off-road Vehicle
- Camper Van
- Compact Car
- Motorcycle
- Electric SUV

Let $v_i$ be the Value (profit) and $w_i$ be the Weight (stock usage) for each product $i$, as given below:

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

The total inventory capacity is:

$\text{Capacity} = 765$

The mathematical model is:

**Objective:**
\[
\max \quad 2524x_{\text{Sedan}} + 4614x_{\text{SUV}} + 8416x_{\text{Truck}} + 5917x_{\text{Convertible}} + 9048x_{\text{Minivan}} + 1140x_{\text{Coupe}} + 8962x_{\text{Hatchback}} + 1888x_{\text{Station Wagon}} + 8487x_{\text{Electric Car}} + 4425x_{\text{Hybrid Car}} + 4717x_{\text{Luxury Sedan}} + 4210x_{\text{Sports Car}} + 1226x_{\text{Crossover}} + 7400x_{\text{Diesel Truck}} + 4639x_{\text{Compact SUV}} + 7712x_{\text{Luxury SUV}} + 3299x_{\text{Cargo Van}} + 9895x_{\text{Pickup Truck}} + 4496x_{\text{Roadster}} + 4526x_{\text{Muscle Car}} + 5688x_{\text{Off-road Vehicle}} + 3007x_{\text{Camper Van}} + 3623x_{\text{Compact Car}} + 8474x_{\text{Motorcycle}} + 8372x_{\text{Electric SUV}}
\]

**Subject to:**
\[
99x_{\text{Sedan}} + 55x_{\text{SUV}} + 75x_{\text{Truck}} + 94x_{\text{Convertible}} + 80x_{\text{Minivan}} + 82x_{\text{Coupe}} + 71x_{\text{Hatchback}} + 100x_{\text{Station Wagon}} + 28x_{\text{Electric Car}} + 93x_{\text{Hybrid Car}} + 84x_{\text{Luxury Sedan}} + 83x_{\text{Sports Car}} + 62x_{\text{Crossover}} + 90x_{\text{Diesel Truck}} + 99x_{\text{Compact SUV}} + 96x_{\text{Luxury SUV}} + 21x_{\text{Cargo Van}} + 39x_{\text{Pickup Truck}} + 99x_{\text{Roadster}} + 81x_{\text{Muscle Car}} + 6x_{\text{Off-road Vehicle}} + 58x_{\text{Camper Van}} + 37x_{\text{Compact Car}} + 15x_{\text{Motorcycle}} + 37x_{\text{Electric SUV}} \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{all ProductName above}\}
\]