Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types listed in the ProductName column.

Parameters:
- $p_i$: Value (profit) for vehicle type $i$
- $w_i$: Weight for vehicle type $i$
- $C$: Capacity (overall stock limit)

Data:

Capacity:
- $C = 765$

Products:

| ProductName         | Value ($p_i$) | Weight ($w_i$) |
|---------------------|--------------|---------------|
| Sedan               | 2524         | 99            |
| SUV                 | 4614         | 55            |
| Truck               | 8416         | 75            |
| Convertible         | 5917         | 94            |
| Minivan             | 9048         | 80            |
| Coupe               | 1140         | 82            |
| Hatchback           | 8962         | 71            |
| Station Wagon       | 1888         | 100           |
| Electric Car        | 8487         | 28            |
| Hybrid Car          | 4425         | 93            |
| Luxury Sedan        | 4717         | 84            |
| Sports Car          | 4210         | 83            |
| Crossover           | 1226         | 62            |
| Diesel Truck        | 7400         | 90            |
| Compact SUV         | 4639         | 99            |
| Luxury SUV          | 7712         | 96            |
| Cargo Van           | 3299         | 21            |
| Pickup Truck        | 9895         | 39            |
| Roadster            | 4496         | 99            |
| Muscle Car          | 4526         | 81            |
| Off-road Vehicle    | 5688         | 6             |
| Camper Van          | 3007         | 58            |
| Compact Car         | 3623         | 37            |
| Motorcycle          | 8474         | 15            |
| Electric SUV        | 8372         | 37            |

Mathematical Model:

Objective:
$$
\max \sum_{i} p_i x_i
$$

Subject to:
$$
\sum_{i} w_i x_i \leq 765
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

Where:
- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = Value for vehicle type $i$ (see table above)
- $w_i$ = Weight for vehicle type $i$ (see table above)
- $765$ = overall inventory capacity

All vehicle types from the ProductName column are included, with their corresponding Value and Weight. The objective is to maximize total profit while ensuring the total weight of ordered vehicles does not exceed the overall capacity.