Let $x_{ij}$ be the number of units of product $j$ (ProductName) placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, indexed by ShelfID:
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Let $P$ be the set of products, indexed by ProductName:
  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$
- For each shelf $i \in S$, let $C_i$ be its capacity:
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
- For each product $j \in P$, let $v_j$ be its value and $w_j$ its weight:

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly, the constraints for each shelf are:**

- Shelf 1: $\sum_{j \in P} w_j x_{1j} \leq 5.0$
- Shelf 2: $\sum_{j \in P} w_j x_{2j} \leq 7.0$
- Shelf 3: $\sum_{j \in P} w_j x_{3j} \leq 6.0$
- Shelf 4: $\sum_{j \in P} w_j x_{4j} \leq 8.0$
- Shelf 5: $\sum_{j \in P} w_j x_{5j} \leq 5.5$
- Shelf 6: $\sum_{j \in P} w_j x_{6j} \leq 9.0$
- Shelf 7: $\sum_{j \in P} w_j x_{7j} \leq 6.5$
- Shelf 8: $\sum_{j \in P} w_j x_{8j} \leq 7.5$
- Shelf 9: $\sum_{j \in P} w_j x_{9j} \leq 8.2$
- Shelf 10: $\sum_{j \in P} w_j x_{10j} \leq 5.7$

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$ = value of product $j$ (see table above)
- $w_j$ = weight of product $j$ (see table above)
- $C_i$ = capacity of shelf $i$ (see above)

---

**All variables and parameters use the exact identifiers and values as retrieved.**