#### Sets and Indices

Let $I$ be the set of vehicle types, indexed by $i$, with the following members (in source order):

| ProductName         |
|---------------------|
| Sedan               |
| SUV                 |
| Truck               |
| Convertible         |
| Minivan             |
| Coupe               |
| Hatchback           |
| Station Wagon       |
| Electric Car        |
| Hybrid Car          |
| Luxury Sedan        |
| Sports Car          |
| Crossover           |
| Diesel Truck        |
| Compact SUV         |
| Luxury SUV          |
| Cargo Van           |
| Pickup Truck        |
| Roadster            |
| Muscle Car          |
| Off-road Vehicle    |
| Camper Van          |
| Compact Car         |
| Motorcycle          |
| Electric SUV        |

#### Parameters

- $p_i$: Profit per unit of vehicle $i$ (from Value column)
- $w_i$: Weight (inventory space required) per unit of vehicle $i$ (from Weight column)
- $C$: Total inventory capacity (from Capacity column)

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

Total inventory capacity: $C = 765$

#### Decision Variables

- $x_i$: Number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**

\[
\sum_{i \in I} w_i x_i \leq C
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

#### Parameter Table (source order)

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

Total inventory capacity: $C = 765$