Let $x_{ij}$ be the number of units of product $j$ placed on display $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Displays (from capacity.csv, in order):

  1. ShelfID 1, Capacity = 5
  2. ShelfID 2, Capacity = 7
  3. ShelfID 3, Capacity = 6
  4. ShelfID 4, Capacity = 8
  5. ShelfID 5, Capacity = 5.5
  6. ShelfID 6, Capacity = 9
  7. ShelfID 7, Capacity = 6.5
  8. ShelfID 8, Capacity = 7.5
  9. ShelfID 9, Capacity = 8.2
  10. ShelfID 10, Capacity = 5.7

- Products (from products.csv, in order):

  1. Smartphone, Weight = 1, Value = 200
  2. Laptop, Weight = 5, Value = 1500
  3. Headphones, Weight = 0.5, Value = 100
  4. Camera, Weight = 2, Value = 800
  5. Smartwatch, Weight = 0.3, Value = 250
  6. Tablet, Weight = 1.5, Value = 600
  7. Bluetooth Speaker, Weight = 1, Value = 150
  8. Keyboard, Weight = 0.8, Value = 80
  9. Mouse, Weight = 0.2, Value = 50
  10. Monitor, Weight = 3, Value = 300
  11. Printer, Weight = 4, Value = 400
  12. External Hard Drive, Weight = 0.5, Value = 120
  13. Router, Weight = 0.3, Value = 60
  14. Power Bank, Weight = 0.4, Value = 40
  15. Memory Card, Weight = 0.05, Value = 30
  16. USB Flash Drive, Weight = 0.02, Value = 25
  17. Smart Home Hub, Weight = 0.6, Value = 100
  18. Gaming Console, Weight = 4, Value = 500
  19. Fitness Tracker, Weight = 0.2, Value = 90
  20. E-Reader, Weight = 0.5, Value = 180

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$

where $v_j$ is the Value of product $j$ as listed above.

**Constraints:**

1. **Display Capacity Constraints (for each display $i$):**

   For $i=1$ (ShelfID 1, Capacity 5):

   $$
   \sum_{j=1}^{20} w_j\, x_{1j} \leq 5
   $$

   For $i=2$ (ShelfID 2, Capacity 7):

   $$
   \sum_{j=1}^{20} w_j\, x_{2j} \leq 7
   $$

   For $i=3$ (ShelfID 3, Capacity 6):

   $$
   \sum_{j=1}^{20} w_j\, x_{3j} \leq 6
   $$

   For $i=4$ (ShelfID 4, Capacity 8):

   $$
   \sum_{j=1}^{20} w_j\, x_{4j} \leq 8
   $$

   For $i=5$ (ShelfID 5, Capacity 5.5):

   $$
   \sum_{j=1}^{20} w_j\, x_{5j} \leq 5.5
   $$

   For $i=6$ (ShelfID 6, Capacity 9):

   $$
   \sum_{j=1}^{20} w_j\, x_{6j} \leq 9
   $$

   For $i=7$ (ShelfID 7, Capacity 6.5):

   $$
   \sum_{j=1}^{20} w_j\, x_{7j} \leq 6.5
   $$

   For $i=8$ (ShelfID 8, Capacity 7.5):

   $$
   \sum_{j=1}^{20} w_j\, x_{8j} \leq 7.5
   $$

   For $i=9$ (ShelfID 9, Capacity 8.2):

   $$
   \sum_{j=1}^{20} w_j\, x_{9j} \leq 8.2
   $$

   For $i=10$ (ShelfID 10, Capacity 5.7):

   $$
   \sum_{j=1}^{20} w_j\, x_{10j} \leq 5.7
   $$

   where $w_j$ is the Weight of product $j$ as listed above.

2. **Minimum Quantity of First Product (Smartphone) Across All Displays:**

$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

---

**Parameter Table (for reference):**

| $j$ | ProductName             | $w_j$ (Weight) | $v_j$ (Value) |
|-----|------------------------|----------------|---------------|
| 1   | Smartphone             | 1              | 200           |
| 2   | Laptop                 | 5              | 1500          |
| 3   | Headphones             | 0.5            | 100           |
| 4   | Camera                 | 2              | 800           |
| 5   | Smartwatch             | 0.3            | 250           |
| 6   | Tablet                 | 1.5            | 600           |
| 7   | Bluetooth Speaker      | 1              | 150           |
| 8   | Keyboard               | 0.8            | 80            |
| 9   | Mouse                  | 0.2            | 50            |
| 10  | Monitor                | 3              | 300           |
| 11  | Printer                | 4              | 400           |
| 12  | External Hard Drive    | 0.5            | 120           |
| 13  | Router                 | 0.3            | 60            |
| 14  | Power Bank             | 0.4            | 40            |
| 15  | Memory Card            | 0.05           | 30            |
| 16  | USB Flash Drive        | 0.02           | 25            |
| 17  | Smart Home Hub         | 0.6            | 100           |
| 18  | Gaming Console         | 4              | 500           |
| 19  | Fitness Tracker        | 0.2            | 90            |
| 20  | E-Reader               | 0.5            | 180           |

| $i$ | ShelfID | $c_i$ (Capacity) |
|-----|---------|------------------|
| 1   | 1       | 5                |
| 2   | 2       | 7                |
| 3   | 3       | 6                |
| 4   | 4       | 8                |
| 5   | 5       | 5.5              |
| 6   | 6       | 9                |
| 7   | 7       | 6.5              |
| 8   | 8       | 7.5              |
| 9   | 9       | 8.2              |
| 10  | 10      | 5.7              |