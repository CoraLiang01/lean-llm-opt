Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID from capacity.csv) and $j$ indexes the products in the order given in products.csv.

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$ (from products.csv). Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

**Product list (in order):**
1. Smartphone
2. Laptop
3. Headphones
4. Camera
5. Smartwatch
6. Tablet
7. Bluetooth Speaker
8. Keyboard
9. Mouse
10. Monitor
11. Printer
12. External Hard Drive
13. Router
14. Power Bank
15. Memory Card
16. USB Flash Drive
17. Smart Home Hub
18. Gaming Console
19. Fitness Tracker
20. E-Reader

**Shelf list (in order):**
1. ShelfID 1, $C_1 = 5.0$
2. ShelfID 2, $C_2 = 7.0$
3. ShelfID 3, $C_3 = 6.0$
4. ShelfID 4, $C_4 = 8.0$
5. ShelfID 5, $C_5 = 5.5$
6. ShelfID 6, $C_6 = 9.0$
7. ShelfID 7, $C_7 = 6.5$
8. ShelfID 8, $C_8 = 7.5$
9. ShelfID 9, $C_9 = 8.2$
10. ShelfID 10, $C_{10} = 5.7$

**Product values and weights:**

| $j$ | Product Name            | $v_j$ | $w_j$  |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1.0    |
| 2   | Laptop                 | 1500  | 5.0    |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2.0    |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1.0    |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3.0    |
| 11  | Printer                | 400   | 4.0    |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4.0    |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

---

### Mathematical Model

**Decision variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$

**Subject to:**

1. **Shelf capacity constraints:**
   $$
   \sum_{j=1}^{20} w_j\, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
   $$
   That is, for each shelf:
   - ShelfID 1: $\sum_{j=1}^{20} w_j\, x_{1j} \leq 5.0$
   - ShelfID 2: $\sum_{j=1}^{20} w_j\, x_{2j} \leq 7.0$
   - ShelfID 3: $\sum_{j=1}^{20} w_j\, x_{3j} \leq 6.0$
   - ShelfID 4: $\sum_{j=1}^{20} w_j\, x_{4j} \leq 8.0$
   - ShelfID 5: $\sum_{j=1}^{20} w_j\, x_{5j} \leq 5.5$
   - ShelfID 6: $\sum_{j=1}^{20} w_j\, x_{6j} \leq 9.0$
   - ShelfID 7: $\sum_{j=1}^{20} w_j\, x_{7j} \leq 6.5$
   - ShelfID 8: $\sum_{j=1}^{20} w_j\, x_{8j} \leq 7.5$
   - ShelfID 9: $\sum_{j=1}^{20} w_j\, x_{9j} \leq 8.2$
   - ShelfID 10: $\sum_{j=1}^{20} w_j\, x_{10j} \leq 5.7$

2. **Minimum total quantity of the first product (Smartphone):**
   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   $$

---

**All coefficients and identifiers are as retrieved and in original order.**