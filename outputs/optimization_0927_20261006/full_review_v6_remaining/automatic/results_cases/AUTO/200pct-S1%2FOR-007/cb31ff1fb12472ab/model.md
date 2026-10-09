Let $I$ be the set of vehicle types, indexed by $i$, with ProductName as identifier. Let $x_i$ be the number of vehicles of type $i$ to order per day (decision variable, nonnegative integer).

Parameters (from products.csv, in source order):

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

From capacity.csv:

- Total inventory capacity: $765$

Model:

Maximize total profit:
$$
\max \sum_{i \in I} \text{Value}_i \cdot x_i
$$

Subject to the inventory capacity constraint:
$$
\sum_{i \in I} \text{Weight}_i \cdot x_i \leq 765
$$

Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Where:

- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $\text{Value}_i$ = profit per unit of vehicle type $i$ (from products.csv)
- $\text{Weight}_i$ = inventory space required per unit of vehicle type $i$ (from products.csv)
- $765$ = total inventory capacity (from capacity.csv)

All data and identifiers are preserved in source order.