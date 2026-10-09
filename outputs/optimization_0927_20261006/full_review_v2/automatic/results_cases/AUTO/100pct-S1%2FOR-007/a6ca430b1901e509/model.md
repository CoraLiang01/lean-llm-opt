Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values as given in products.csv.

**Parameters:**

- For each vehicle type $i$:
    - $p_i$ = Value (profit per unit)
    - $w_i$ = Weight (stock space per unit)
- $C$ = 765 (overall inventory capacity, from capacity.csv)

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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i} p_i x_i
\]
where $p_i$ and $x_i$ are as above.

**Subject to:**

\[
\sum_{i} w_i x_i \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Where:**

- $p_i$, $w_i$ are as listed above for each ProductName.
- $x_i$ is the number of units of vehicle type $i$ to order per day.

**All variables and coefficients are as retrieved and in original source order.**