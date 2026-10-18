##### Sets and Indices

- Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of displays (ShelfID).
- Let $J = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ be the set of products.
- Let $j^* =$ Smartphone (the first product in the list).

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

| $j$                   | Value $v_j$ | Weight $w_j$ |
|-----------------------|-------------|--------------|
| Smartphone            | 200         | 1.0          |
| Laptop                | 1500        | 5.0          |
| Headphones            | 100         | 0.5          |
| Camera                | 800         | 2.0          |
| Smartwatch            | 250         | 0.3          |
| Tablet                | 600         | 1.5          |
| Bluetooth Speaker     | 150         | 1.0          |
| Keyboard              | 80          | 0.8          |
| Mouse                 | 50          | 0.2          |
| Monitor               | 300         | 3.0          |
| Printer               | 400         | 4.0          |
| External Hard Drive   | 120         | 0.5          |
| Router                | 60          | 0.3          |
| Power Bank            | 40          | 0.4          |
| Memory Card           | 30          | 0.05         |
| USB Flash Drive       | 25          | 0.02         |
| Smart Home Hub        | 100         | 0.6          |
| Gaming Console        | 500         | 4.0          |
| Fitness Tracker       | 90          | 0.2          |
| E-Reader              | 180         | 0.5          |

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j \in J$ placed on display $i \in I$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Display capacity constraints** (for each display $i$):

   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Minimum total quantity for the first product (Smartphone):**

   \[
   \sum_{i \in I} x_{i, \text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and integrality:**

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

###### Retrieved Information

- Displays $I = \{1,2,3,4,5,6,7,8,9,10\}$ with capacities:
  - $C_1 = 5.0$, $C_2 = 7.0$, $C_3 = 6.0$, $C_4 = 8.0$, $C_5 = 5.5$, $C_6 = 9.0$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$
- Products $J$ with values and weights as listed above.
- The first product is Smartphone.

This model maximizes the total value of products placed on displays, subject to display capacity and minimum Smartphone quantity constraints.