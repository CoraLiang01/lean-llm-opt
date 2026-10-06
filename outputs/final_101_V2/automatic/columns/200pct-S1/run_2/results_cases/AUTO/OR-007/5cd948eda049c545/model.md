Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the ProductName column of products.csv.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value for vehicle type $i$ from products.csv.

Subject to:

Inventory Capacity Constraint:
$$
\sum_{i} w_i x_i \leq 765
$$
where $w_i$ is the Weight for vehicle type $i$ from products.csv, and 765 is the Capacity from capacity.csv.

Non-negativity and Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:

- Vehicle types $i$ and their parameters (in source order):

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

- Capacity: 765

Decision variables $x_i$ are nonnegative integers for each vehicle type $i$ listed above.