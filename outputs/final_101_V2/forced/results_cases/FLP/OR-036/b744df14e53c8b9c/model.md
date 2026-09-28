##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ to order daily, for each $i \in P$ (where $P$ is the set of all vehicle types listed below).

##### Parameters

Let $v_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $w_i$ be the inventory weight for vehicle type $i$.

Let $C$ be the total inventory capacity.

From the data:

$C = 1576$

$P = \{$
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
$\}$

The parameters for each $i \in P$ are:

| $i$                | $v_i$ | $w_i$ |
|--------------------|-------|-------|
| Sedan              | 1752  | 15    |
| SUV                | 1856  | 87    |
| Truck              | 8372  | 36    |
| Convertible        | 6168  | 30    |
| Minivan            | 9681  | 33    |
| Coupe              | 8062  | 72    |
| Hatchback          | 3895  | 75    |
| Station Wagon      | 3254  | 71    |
| Electric Car       | 1701  | 51    |
| Hybrid Car         | 6799  | 21    |
| Luxury Sedan       | 2724  | 97    |
| Sports Car         | 6304  | 52    |
| Crossover          | 3255  | 25    |
| Diesel Truck       | 1923  | 15    |
| Compact SUV        | 4103  | 54    |
| Luxury SUV         | 4429  | 57    |
| Cargo Van          | 2663  | 18    |
| Pickup Truck       | 1691  | 69    |
| Roadster           | 5632  | 26    |
| Muscle Car         | 4793  | 38    |
| Off-road Vehicle   | 1343  | 31    |
| Camper Van         | 9124  | 74    |
| Compact Car        | 3652  | 82    |
| Motorcycle         | 8842  | 49    |
| Electric SUV       | 9176  | 64    |

##### Objective Function

\[
\max \sum_{i \in P} v_i x_i
\]

##### Constraints

1. Inventory capacity:
   \[
   \sum_{i \in P} w_i x_i \leq C
   \]
2. Integer and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in P
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\max\quad & \sum_{i \in P} v_i x_i \\
\text{s.t.}\quad & \sum_{i \in P} w_i x_i \leq 1576 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in P
\end{align*}
\]

Where $P$ and all parameters $v_i$, $w_i$ are as listed above.