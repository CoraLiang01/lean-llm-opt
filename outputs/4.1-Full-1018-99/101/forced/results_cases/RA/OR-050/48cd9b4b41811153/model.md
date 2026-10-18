Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$ (ShelfID from capacity.csv) and $j$ indexes the products in the order given in products.csv.

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$ (from products.csv). Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

Define the following sets and parameters:

- Shelves (displays): $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$
- Products $j$ in order:
  1. Smartphone ($v_1 = 200$, $w_1 = 1.0$)
  2. Laptop ($v_2 = 1500$, $w_2 = 5.0$)
  3. Headphones ($v_3 = 100$, $w_3 = 0.5$)
  4. Camera ($v_4 = 800$, $w_4 = 2.0$)
  5. Smartwatch ($v_5 = 250$, $w_5 = 0.3$)
  6. Tablet ($v_6 = 600$, $w_6 = 1.5$)
  7. Bluetooth Speaker ($v_7 = 150$, $w_7 = 1.0$)
  8. Keyboard ($v_8 = 80$, $w_8 = 0.8$)
  9. Mouse ($v_9 = 50$, $w_9 = 0.2$)
  10. Monitor ($v_{10} = 300$, $w_{10} = 3.0$)
  11. Printer ($v_{11} = 400$, $w_{11} = 4.0$)
  12. External Hard Drive ($v_{12} = 120$, $w_{12} = 0.5$)
  13. Router ($v_{13} = 60$, $w_{13} = 0.3$)
  14. Power Bank ($v_{14} = 40$, $w_{14} = 0.4$)
  15. Memory Card ($v_{15} = 30$, $w_{15} = 0.05$)
  16. USB Flash Drive ($v_{16} = 25$, $w_{16} = 0.02$)
  17. Smart Home Hub ($v_{17} = 100$, $w_{17} = 0.6$)
  18. Gaming Console ($v_{18} = 500$, $w_{18} = 4.0$)
  19. Fitness Tracker ($v_{19} = 90$, $w_{19} = 0.2$)
  20. E-Reader ($v_{20} = 180$, $w_{20} = 0.5$)

Shelf capacities:
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

The mathematical model is:

**Decision variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

**Subject to:**

1. **Shelf capacity constraints:**
   $$
   \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
   $$
   That is, for each shelf:
   \begin{align*}
   &\sum_{j=1}^{20} w_j x_{1j} \leq 5.0 \\
   &\sum_{j=1}^{20} w_j x_{2j} \leq 7.0 \\
   &\sum_{j=1}^{20} w_j x_{3j} \leq 6.0 \\
   &\sum_{j=1}^{20} w_j x_{4j} \leq 8.0 \\
   &\sum_{j=1}^{20} w_j x_{5j} \leq 5.5 \\
   &\sum_{j=1}^{20} w_j x_{6j} \leq 9.0 \\
   &\sum_{j=1}^{20} w_j x_{7j} \leq 6.5 \\
   &\sum_{j=1}^{20} w_j x_{8j} \leq 7.5 \\
   &\sum_{j=1}^{20} w_j x_{9j} \leq 8.2 \\
   &\sum_{j=1}^{20} w_j x_{10j} \leq 5.7 \\
   \end{align*}

2. **Minimum total quantity of the first product (Smartphone):**
   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   $$

**Parameter Table (in source order):**

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

| ProductName            | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1.0    |
| Laptop                 | 1500  | 5.0    |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2.0    |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1.0    |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3.0    |
| Printer                | 400   | 4.0    |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4.0    |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |