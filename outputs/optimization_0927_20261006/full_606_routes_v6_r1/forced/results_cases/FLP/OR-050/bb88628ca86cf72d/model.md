##### Sets and Indices

- Displays: $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Products: $J = \{$
  1: Smartphone,
  2: Laptop,
  3: Headphones,
  4: Camera,
  5: Smartwatch,
  6: Tablet,
  7: Bluetooth Speaker,
  8: Keyboard,
  9: Mouse,
  10: Monitor,
  11: Printer,
  12: External Hard Drive,
  13: Router,
  14: Power Bank,
  15: Memory Card,
  16: USB Flash Drive,
  17: Smart Home Hub,
  18: Gaming Console,
  19: Fitness Tracker,
  20: E-Reader
$\}$

##### Parameters

- Display capacities:
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

| $j$ | Product Name           | $v_j$ (Value) | $w_j$ (Weight) |
|-----|-----------------------|---------------|----------------|
| 1   | Smartphone            | 200           | 1.0            |
| 2   | Laptop                | 1500          | 5.0            |
| 3   | Headphones            | 100           | 0.5            |
| 4   | Camera                | 800           | 2.0            |
| 5   | Smartwatch            | 250           | 0.3            |
| 6   | Tablet                | 600           | 1.5            |
| 7   | Bluetooth Speaker     | 150           | 1.0            |
| 8   | Keyboard              | 80            | 0.8            |
| 9   | Mouse                 | 50            | 0.2            |
| 10  | Monitor               | 300           | 3.0            |
| 11  | Printer               | 400           | 4.0            |
| 12  | External Hard Drive   | 120           | 0.5            |
| 13  | Router                | 60            | 0.3            |
| 14  | Power Bank            | 40            | 0.4            |
| 15  | Memory Card           | 30            | 0.05           |
| 16  | USB Flash Drive       | 25            | 0.02           |
| 17  | Smart Home Hub        | 100           | 0.6            |
| 18  | Gaming Console        | 500           | 4.0            |
| 19  | Fitness Tracker       | 90            | 0.2            |
| 20  | E-Reader              | 180           | 0.5            |

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$.

##### Objective Function

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

##### Constraints

1. **Display capacity constraints** (for each display $i$):

   \[
   \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\}
   \]

2. **Minimum total quantity of the first product (Smartphone) across all displays:**

   \[
   \sum_{i=1}^{10} x_{i,1} \geq 5
   \]

3. **Nonnegativity and integrality:**

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   \]

##### All Parameters (retrieved):

- Displays and capacities:
  - 1: 5.0
  - 2: 7.0
  - 3: 6.0
  - 4: 8.0
  - 5: 5.5
  - 6: 9.0
  - 7: 6.5
  - 8: 7.5
  - 9: 8.2
  - 10: 5.7

- Products, values, and weights:
  - 1: Smartphone, 200, 1.0
  - 2: Laptop, 1500, 5.0
  - 3: Headphones, 100, 0.5
  - 4: Camera, 800, 2.0
  - 5: Smartwatch, 250, 0.3
  - 6: Tablet, 600, 1.5
  - 7: Bluetooth Speaker, 150, 1.0
  - 8: Keyboard, 80, 0.8
  - 9: Mouse, 50, 0.2
  - 10: Monitor, 300, 3.0
  - 11: Printer, 400, 4.0
  - 12: External Hard Drive, 120, 0.5
  - 13: Router, 60, 0.3
  - 14: Power Bank, 40, 0.4
  - 15: Memory Card, 30, 0.05
  - 16: USB Flash Drive, 25, 0.02
  - 17: Smart Home Hub, 100, 0.6
  - 18: Gaming Console, 500, 4.0
  - 19: Fitness Tracker, 90, 0.2
  - 20: E-Reader, 180, 0.5