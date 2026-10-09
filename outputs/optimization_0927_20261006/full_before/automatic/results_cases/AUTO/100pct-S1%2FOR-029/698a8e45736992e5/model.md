Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

Let $S$ be the set of shelves (from "capacity.csv") and $P$ the set of products (from "products.csv"). Let $c_i$ be the capacity of shelf $i$, $v_j$ the value of product $j$, and $w_j$ the weight of product $j$.

Define:
- Shelves $i \in \{$1, 2, 3, 4, 5, 6, 7, 8, 9, 10$\}$, with capacities as below.
- Products $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$, with values and weights as below.

#### Parameters

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

| Product Name           | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1      |
| Laptop                 | 1500  | 5      |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2      |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1      |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3      |
| Printer                | 400   | 4      |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4      |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |

#### Mathematical Model

Objective:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
$$

Subject to:

Capacity constraints for each shelf:
$$
\sum_{j \in P} w_j \, x_{ij} \leq c_i \qquad \forall i \in S
$$

Minimum total quantity for the first product ("Smartphone"):
$$
\sum_{i \in S} x_{i,\text{Smartphone}} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
$$

Where:
- $S = \{1,2,3,4,5,6,7,8,9,10\}$
- $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$
- $c_i$ is the "Capacity" of shelf $i$ as above
- $v_j$ is the "Value" of product $j$ as above
- $w_j$ is the "Weight" of product $j$ as above