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

Let $p_i$ be the Value (profit) for each product $i$, and $w_i$ be the Weight (stock space required) for each product $i$. The total available inventory capacity is $765$ units.

The complete mathematical model is:

Objective:
\[
\max \sum_{i} p_i x_i
\]

where the data for $p_i$ and $w_i$ are:

| ProductName         | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------|:-------------:|:--------------:|
| Sedan               | 2524          | 99             |
| SUV                 | 4614          | 55             |
| Truck               | 8416          | 75             |
| Convertible         | 5917          | 94             |
| Minivan             | 9048          | 80             |
| Coupe               | 1140          | 82             |
| Hatchback           | 8962          | 71             |
| Station Wagon       | 1888          | 100            |
| Electric Car        | 8487          | 28             |
| Hybrid Car          | 4425          | 93             |
| Luxury Sedan        | 4717          | 84             |
| Sports Car          | 4210          | 83             |
| Crossover           | 1226          | 62             |
| Diesel Truck        | 7400          | 90             |
| Compact SUV         | 4639          | 99             |
| Luxury SUV          | 7712          | 96             |
| Cargo Van           | 3299          | 21             |
| Pickup Truck        | 9895          | 39             |
| Roadster            | 4496          | 99             |
| Muscle Car          | 4526          | 81             |
| Off-road Vehicle    | 5688          | 6              |
| Camper Van          | 3007          | 58             |
| Compact Car         | 3623          | 37             |
| Motorcycle          | 8474          | 15             |
| Electric SUV        | 8372          | 37             |

Subject to:

Capacity constraint:
\[
\sum_{i} w_i x_i \leq 765
\]

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where:
- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $p_i$ = profit per vehicle of type $i$ (Value column)
- $w_i$ = stock space required per vehicle of type $i$ (Weight column)
- $765$ = total available inventory capacity

All coefficients and identifiers are as retrieved and preserved in source order.