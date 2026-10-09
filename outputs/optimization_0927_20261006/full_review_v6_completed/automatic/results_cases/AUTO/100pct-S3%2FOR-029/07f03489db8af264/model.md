Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from the capacity.csv (in source order: 1, 2, ..., 10), and $j$ indexes ProductName from products.csv (in source order: Smartphone, Laptop, Headphones, ..., E-Reader). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $v_j$: Value of product $j$ (from "Value" in products.csv)
- $w_j$: Weight of product $j$ (from "Weight" in products.csv)
- $C_i$: Capacity of display $i$ (from "Capacity" in capacity.csv)

**Data (in source order):**

- Displays (ShelfID, Capacity):

    1. 1, 5
    2. 2, 7
    3. 3, 6
    4. 4, 8
    5. 5, 5.5
    6. 6, 9
    7. 7, 6.5
    8. 8, 7.5
    9. 9, 8.2
    10. 10, 5.7

- Products (ProductName, Value, Weight):

    1. Smartphone, 200, 1
    2. Laptop, 1500, 5
    3. Headphones, 100, 0.5
    4. Camera, 800, 2
    5. Smartwatch, 250, 0.3
    6. Tablet, 600, 1.5
    7. Bluetooth Speaker, 150, 1
    8. Keyboard, 80, 0.8
    9. Mouse, 50, 0.2
    10. Monitor, 300, 3
    11. Printer, 400, 4
    12. External Hard Drive, 120, 0.5
    13. Router, 60, 0.3
    14. Power Bank, 40, 0.4
    15. Memory Card, 30, 0.05
    16. USB Flash Drive, 25, 0.02
    17. Smart Home Hub, 100, 0.6
    18. Gaming Console, 500, 4
    19. Fitness Tracker, 90, 0.2
    20. E-Reader, 180, 0.5

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$

**Subject to:**

1. **Display Capacity Constraints (for each display $i$):**

   $$
   \sum_{j=1}^{20} w_j\, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
   $$

   That is, for each shelf:

   - Shelf 1: $\sum_{j=1}^{20} w_j\, x_{1j} \leq 5$
   - Shelf 2: $\sum_{j=1}^{20} w_j\, x_{2j} \leq 7$
   - Shelf 3: $\sum_{j=1}^{20} w_j\, x_{3j} \leq 6$
   - Shelf 4: $\sum_{j=1}^{20} w_j\, x_{4j} \leq 8$
   - Shelf 5: $\sum_{j=1}^{20} w_j\, x_{5j} \leq 5.5$
   - Shelf 6: $\sum_{j=1}^{20} w_j\, x_{6j} \leq 9$
   - Shelf 7: $\sum_{j=1}^{20} w_j\, x_{7j} \leq 6.5$
   - Shelf 8: $\sum_{j=1}^{20} w_j\, x_{8j} \leq 7.5$
   - Shelf 9: $\sum_{j=1}^{20} w_j\, x_{9j} \leq 8.2$
   - Shelf 10: $\sum_{j=1}^{20} w_j\, x_{10j} \leq 5.7$

2. **Minimum Quantity of First Product (Smartphone) Across All Displays:**

   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and Integrality:**

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   $$

---

**Where:**

- $x_{ij}$: Number of units of product $j$ (see above order) placed on display $i$ (see above order)
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $C_i$: Capacity of display $i$ (see above)

---

**All data and indices are in the original source order as retrieved.**