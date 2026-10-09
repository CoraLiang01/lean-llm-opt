Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves, and $J$ be the set of products:

$J = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

Let $c_i$ be the capacity of shelf $i$, $v_j$ the value of product $j$, and $w_j$ the weight of product $j$.

Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ be the number of units of product $j$ placed on shelf $i$.

Parameters (from retrieved data):

Shelf capacities:
- $c_1 = 5.0$
- $c_2 = 7.0$
- $c_3 = 6.0$
- $c_4 = 8.0$
- $c_5 = 5.5$
- $c_6 = 9.0$
- $c_7 = 6.5$
- $c_8 = 7.5$
- $c_9 = 8.2$
- $c_{10} = 5.7$

Product values and weights:

| $j$                  | $v_j$ | $w_j$  |
|----------------------|-------|--------|
| Smartphone           | 200   | 1.0    |
| Laptop               | 1500  | 5.0    |
| Headphones           | 100   | 0.5    |
| Camera               | 800   | 2.0    |
| Smartwatch           | 250   | 0.3    |
| Tablet               | 600   | 1.5    |
| Bluetooth Speaker    | 150   | 1.0    |
| Keyboard             | 80    | 0.8    |
| Mouse                | 50    | 0.2    |
| Monitor              | 300   | 3.0    |
| Printer              | 400   | 4.0    |
| External Hard Drive  | 120   | 0.5    |
| Router               | 60    | 0.3    |
| Power Bank           | 40    | 0.4    |
| Memory Card          | 30    | 0.05   |
| USB Flash Drive      | 25    | 0.02   |
| Smart Home Hub       | 100   | 0.6    |
| Gaming Console       | 500   | 4.0    |
| Fitness Tracker      | 90    | 0.2    |
| E-Reader             | 180   | 0.5    |

Model:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j \in J} v_j x_{ij}
$$

Subject to shelf capacity constraints:
$$
\sum_{j \in J} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
$$

Integer nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\; j \in J
$$

All parameters and identifiers are as retrieved above.