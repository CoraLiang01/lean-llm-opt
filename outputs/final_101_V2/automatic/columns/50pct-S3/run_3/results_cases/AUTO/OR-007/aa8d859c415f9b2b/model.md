Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as given by the ProductName column in products.csv.

Objective:
\[
\max \sum_{i} \text{Value}_i \cdot x_i
\]
where $\text{Value}_i$ is the profit from selling one unit of vehicle type $i$.

Subject to:

Capacity constraint:
\[
\sum_{i} \text{Weight}_i \cdot x_i \leq 765
\]
where $\text{Weight}_i$ is the weight (space requirement) of vehicle type $i$, and 765 is the overall inventory capacity from capacity.csv.

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:

- Vehicle types $i$ and their parameters are:

| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Sedan              | 2524  | 99     |
| SUV                | 4614  | 55     |
| Truck              | 8416  | 75     |
| Convertible        | 5917  | 94     |
| Minivan            | 9048  | 80     |
| Coupe              | 1140  | 82     |
| Hatchback          | 8962  | 71     |
| Station Wagon      | 1888  | 100    |
| Electric Car       | 8487  | 28     |
| Hybrid Car         | 4425  | 93     |
| Luxury Sedan       | 4717  | 84     |
| Sports Car         | 4210  | 83     |
| Crossover          | 1226  | 62     |
| Diesel Truck       | 7400  | 90     |
| Compact SUV        | 4639  | 99     |
| Luxury SUV         | 7712  | 96     |
| Cargo Van          | 3299  | 21     |
| Pickup Truck       | 9895  | 39     |
| Roadster           | 4496  | 99     |
| Muscle Car         | 4526  | 81     |
| Off-road Vehicle   | 5688  | 6      |
| Camper Van         | 3007  | 58     |
| Compact Car        | 3623  | 37     |
| Motorcycle         | 8474  | 15     |
| Electric SUV       | 8372  | 37     |

All $x_i$ are nonnegative integers. The objective is to maximize total profit from the ordered vehicles, subject to the total inventory weight not exceeding 765 units.