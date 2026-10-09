##### Sets and Indices

- Let $S = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves, indexed by $i$.
- Let $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ be the set of products, indexed by $j$.

##### Parameters

- Shelf capacities:
  - $C_1 = 5.0$
  - $C_2 = 7.0$
  - $C_3 = 6.0$
  - $C_4 = 8.0$
  - $C_5 = 5.5$
  - $C_6 = 9.0$
  - $C_7 = 6.5$
  - $C_8 = 7.5$
  - $C_9 = 8.2$
  - $C_{10} = 5.7$

- Product values and weights:

| $j$ (Product)            | $v_j$ (Value) | $w_j$ (Weight) |
|------------------------- |:-------------:|:--------------:|
| Smartphone               | 200           | 1.0            |
| Laptop                   | 1500          | 5.0            |
| Headphones               | 100           | 0.5            |
| Camera                   | 800           | 2.0            |
| Smartwatch               | 250           | 0.3            |
| Tablet                   | 600           | 1.5            |
| Bluetooth Speaker        | 150           | 1.0            |
| Keyboard                 | 80            | 0.8            |
| Mouse                    | 50            | 0.2            |
| Monitor                  | 300           | 3.0            |
| Printer                  | 400           | 4.0            |
| External Hard Drive      | 120           | 0.5            |
| Router                   | 60            | 0.3            |
| Power Bank               | 40            | 0.4            |
| Memory Card              | 30            | 0.05           |
| USB Flash Drive          | 25            | 0.02           |
| Smart Home Hub           | 100           | 0.6            |
| Gaming Console           | 500           | 4.0            |
| Fitness Tracker          | 90            | 0.2            |
| E-Reader                 | 180           | 0.5            |

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$ (integer, nonnegative).

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. **Shelf capacity constraints:** For each shelf $i \in S$,
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
   \]

##### Retrieved Information

- Shelves and capacities:
  - $S = \{1,2,3,4,5,6,7,8,9,10\}$
  - $C_1 = 5.0$, $C_2 = 7.0$, $C_3 = 6.0$, $C_4 = 8.0$, $C_5 = 5.5$, $C_6 = 9.0$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$

- Products, values, and weights:
  - Smartphone: $v = 200$, $w = 1.0$
  - Laptop: $v = 1500$, $w = 5.0$
  - Headphones: $v = 100$, $w = 0.5$
  - Camera: $v = 800$, $w = 2.0$
  - Smartwatch: $v = 250$, $w = 0.3$
  - Tablet: $v = 600$, $w = 1.5$
  - Bluetooth Speaker: $v = 150$, $w = 1.0$
  - Keyboard: $v = 80$, $w = 0.8$
  - Mouse: $v = 50$, $w = 0.2$
  - Monitor: $v = 300$, $w = 3.0$
  - Printer: $v = 400$, $w = 4.0$
  - External Hard Drive: $v = 120$, $w = 0.5$
  - Router: $v = 60$, $w = 0.3$
  - Power Bank: $v = 40$, $w = 0.4$
  - Memory Card: $v = 30$, $w = 0.05$
  - USB Flash Drive: $v = 25$, $w = 0.02$
  - Smart Home Hub: $v = 100$, $w = 0.6$
  - Gaming Console: $v = 500$, $w = 4.0$
  - Fitness Tracker: $v = 90$, $w = 0.2$
  - E-Reader: $v = 180$, $w = 0.5$