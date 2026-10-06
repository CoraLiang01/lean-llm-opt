Let $x_{ij}$ be the number of units of product $j$ placed on display $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Parameters:**

- Displays (from capacity.csv, in order):

  1. ShelfID 1, Capacity 5
  2. ShelfID 2, Capacity 7
  3. ShelfID 3, Capacity 6
  4. ShelfID 4, Capacity 8
  5. ShelfID 5, Capacity 5.5
  6. ShelfID 6, Capacity 9
  7. ShelfID 7, Capacity 6.5
  8. ShelfID 8, Capacity 7.5
  9. ShelfID 9, Capacity 8.2
  10. ShelfID 10, Capacity 5.7

- Products (from products.csv, in order):

  1. Smartphone, Value 200, Weight 1
  2. Laptop, Value 1500, Weight 5
  3. Headphones, Value 100, Weight 0.5
  4. Camera, Value 800, Weight 2
  5. Smartwatch, Value 250, Weight 0.3
  6. Tablet, Value 600, Weight 1.5
  7. Bluetooth Speaker, Value 150, Weight 1
  8. Keyboard, Value 80, Weight 0.8
  9. Mouse, Value 50, Weight 0.2
  10. Monitor, Value 300, Weight 3
  11. Printer, Value 400, Weight 4
  12. External Hard Drive, Value 120, Weight 0.5
  13. Router, Value 60, Weight 0.3
  14. Power Bank, Value 40, Weight 0.4
  15. Memory Card, Value 30, Weight 0.05
  16. USB Flash Drive, Value 25, Weight 0.02
  17. Smart Home Hub, Value 100, Weight 0.6
  18. Gaming Console, Value 500, Weight 4
  19. Fitness Tracker, Value 90, Weight 0.2
  20. E-Reader, Value 180, Weight 0.5

---

**Mathematical Model:**

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for $i \in \{1,\ldots,10\}$ (ShelfID), $j \in \{1,\ldots,20\}$ (ProductName)

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ (see above).

**Constraints:**

1. **Display Capacity Constraints (for each display $i$):**

   For each $i \in \{1,\ldots,10\}$ (ShelfID):

   $$
   \sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
   $$

   where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of ShelfID $i$.

   Explicitly, for each shelf:

   - ShelfID 1: $\sum_{j=1}^{20} w_j x_{1j} \leq 5$
   - ShelfID 2: $\sum_{j=1}^{20} w_j x_{2j} \leq 7$
   - ShelfID 3: $\sum_{j=1}^{20} w_j x_{3j} \leq 6$
   - ShelfID 4: $\sum_{j=1}^{20} w_j x_{4j} \leq 8$
   - ShelfID 5: $\sum_{j=1}^{20} w_j x_{5j} \leq 5.5$
   - ShelfID 6: $\sum_{j=1}^{20} w_j x_{6j} \leq 9$
   - ShelfID 7: $\sum_{j=1}^{20} w_j x_{7j} \leq 6.5$
   - ShelfID 8: $\sum_{j=1}^{20} w_j x_{8j} \leq 7.5$
   - ShelfID 9: $\sum_{j=1}^{20} w_j x_{9j} \leq 8.2$
   - ShelfID 10: $\sum_{j=1}^{20} w_j x_{10j} \leq 5.7$

2. **Minimum Quantity of First Product (Smartphone) Across All Displays:**

   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and Integrality:**

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   $$

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on display $i$
- $v_j$ = Value of product $j$ (see above)
- $w_j$ = Weight of product $j$ (see above)
- $C_i$ = Capacity of display $i$ (see above)

All indices, coefficients, and constraints are as retrieved and in original order.