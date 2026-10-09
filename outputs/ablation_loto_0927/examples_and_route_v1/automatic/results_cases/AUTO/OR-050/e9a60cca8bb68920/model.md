Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $S$ be the set of ShelfIDs (displays), indexed by $i$:
$$
S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $P$ be the set of ProductNames, indexed by $j$:
$$
P = \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}
$$

Let $v_j$ be the Value of product $j$ and $w_j$ be the Weight of product $j$ (see table below).
Let $C_i$ be the Capacity of shelf $i$ (see table below).

#### Parameters

| ShelfID ($i$) | Capacity $C_i$ |
|:-------------:|:--------------:|
| 1             | 5.0            |
| 2             | 7.0            |
| 3             | 6.0            |
| 4             | 8.0            |
| 5             | 5.5            |
| 6             | 9.0            |
| 7             | 6.5            |
| 8             | 7.5            |
| 9             | 8.2            |
| 10            | 5.7            |

| ProductName ($j$)         | Value $v_j$ | Weight $w_j$ |
|---------------------------|:-----------:|:------------:|
| Smartphone                | 200         | 1.0          |
| Laptop                    | 1500        | 5.0          |
| Headphones                | 100         | 0.5          |
| Camera                    | 800         | 2.0          |
| Smartwatch                | 250         | 0.3          |
| Tablet                    | 600         | 1.5          |
| Bluetooth Speaker         | 150         | 1.0          |
| Keyboard                  | 80          | 0.8          |
| Mouse                     | 50          | 0.2          |
| Monitor                   | 300         | 3.0          |
| Printer                   | 400         | 4.0          |
| External Hard Drive       | 120         | 0.5          |
| Router                    | 60          | 0.3          |
| Power Bank                | 40          | 0.4          |
| Memory Card               | 30          | 0.05         |
| USB Flash Drive           | 25          | 0.02         |
| Smart Home Hub            | 100         | 0.6          |
| Gaming Console            | 500         | 4.0          |
| Fitness Tracker           | 90          | 0.2          |
| E-Reader                  | 180         | 0.5          |

#### Decision Variables

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \; j \in P
$$

#### Objective Function

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):

$$
\sum_{j \in P} w_j \, x_{ij} \leq C_i \qquad \forall i \in S
$$

2. **Minimum Allocation of First Product ("Smartphone")**:

$$
\sum_{i \in S} x_{i,\text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality**:

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \; j \in P
$$

All parameters and identifiers are as retrieved and preserved in source order.