Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the 25 vehicle types in the table below.

**Objective:**
\[
\max \sum_{i=1}^{25} p_i x_i
\]
where $p_i$ is the Value (profit) of vehicle type $i$.

**Subject to:**

\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]
where $w_i$ is the Weight of vehicle type $i$.

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]

**Data:**

| $i$ | ProductName           | $p_i$ (Value) | $w_i$ (Weight) |
|-----|-----------------------|---------------|----------------|
| 1   | Sedan                 | 2524          | 99             |
| 2   | SUV                   | 4614          | 55             |
| 3   | Truck                 | 8416          | 75             |
| 4   | Convertible           | 5917          | 94             |
| 5   | Minivan               | 9048          | 80             |
| 6   | Coupe                 | 1140          | 82             |
| 7   | Hatchback             | 8962          | 71             |
| 8   | Station Wagon         | 1888          | 100            |
| 9   | Electric Car          | 8487          | 28             |
| 10  | Hybrid Car            | 4425          | 93             |
| 11  | Luxury Sedan          | 4717          | 84             |
| 12  | Sports Car            | 4210          | 83             |
| 13  | Crossover             | 1226          | 62             |
| 14  | Diesel Truck          | 7400          | 90             |
| 15  | Compact SUV           | 4639          | 99             |
| 16  | Luxury SUV            | 7712          | 96             |
| 17  | Cargo Van             | 3299          | 21             |
| 18  | Pickup Truck          | 9895          | 39             |
| 19  | Roadster              | 4496          | 99             |
| 20  | Muscle Car            | 4526          | 81             |
| 21  | Off-road Vehicle      | 5688          | 6              |
| 22  | Camper Van            | 3007          | 58             |
| 23  | Compact Car           | 3623          | 37             |
| 24  | Motorcycle            | 8474          | 15             |
| 25  | Electric SUV          | 8372          | 37             |

**Capacity:**
\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 25
\]