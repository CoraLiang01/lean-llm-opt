Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

Parameters:

- Shelves (from capacity.csv, in order):

    1. ShelfID 1, Capacity = 5.0
    2. ShelfID 2, Capacity = 7.0
    3. ShelfID 3, Capacity = 6.0
    4. ShelfID 4, Capacity = 8.0
    5. ShelfID 5, Capacity = 5.5
    6. ShelfID 6, Capacity = 9.0
    7. ShelfID 7, Capacity = 6.5
    8. ShelfID 8, Capacity = 7.5
    9. ShelfID 9, Capacity = 8.2
    10. ShelfID 10, Capacity = 5.7

- Products (from products.csv, in order):

    1. Smartphone: Value = 200, Weight = 1.0
    2. Laptop: Value = 1500, Weight = 5.0
    3. Headphones: Value = 100, Weight = 0.5
    4. Camera: Value = 800, Weight = 2.0
    5. Smartwatch: Value = 250, Weight = 0.3
    6. Tablet: Value = 600, Weight = 1.5
    7. Bluetooth Speaker: Value = 150, Weight = 1.0
    8. Keyboard: Value = 80, Weight = 0.8
    9. Mouse: Value = 50, Weight = 0.2
    10. Monitor: Value = 300, Weight = 3.0
    11. Printer: Value = 400, Weight = 4.0
    12. External Hard Drive: Value = 120, Weight = 0.5
    13. Router: Value = 60, Weight = 0.3
    14. Power Bank: Value = 40, Weight = 0.4
    15. Memory Card: Value = 30, Weight = 0.05
    16. USB Flash Drive: Value = 25, Weight = 0.02
    17. Smart Home Hub: Value = 100, Weight = 0.6
    18. Gaming Console: Value = 500, Weight = 4.0
    19. Fitness Tracker: Value = 90, Weight = 0.2
    20. E-Reader: Value = 180, Weight = 0.5

Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$ as listed above.

Constraints:

For each shelf $i$ (ShelfID from 1 to 10, with corresponding Capacity $C_i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
$$

where $w_j$ is the Weight of product $j$ as listed above, and $C_i$ is the Capacity of shelf $i$ as listed above.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
$$

Explicitly, the parameters are:

- For $i=1$ (ShelfID 1): $C_1 = 5.0$
- For $i=2$ (ShelfID 2): $C_2 = 7.0$
- For $i=3$ (ShelfID 3): $C_3 = 6.0$
- For $i=4$ (ShelfID 4): $C_4 = 8.0$
- For $i=5$ (ShelfID 5): $C_5 = 5.5$
- For $i=6$ (ShelfID 6): $C_6 = 9.0$
- For $i=7$ (ShelfID 7): $C_7 = 6.5$
- For $i=8$ (ShelfID 8): $C_8 = 7.5$
- For $i=9$ (ShelfID 9): $C_9 = 8.2$
- For $i=10$ (ShelfID 10): $C_{10} = 5.7$

- For $j=1$ (Smartphone): $v_1 = 200$, $w_1 = 1.0$
- For $j=2$ (Laptop): $v_2 = 1500$, $w_2 = 5.0$
- For $j=3$ (Headphones): $v_3 = 100$, $w_3 = 0.5$
- For $j=4$ (Camera): $v_4 = 800$, $w_4 = 2.0$
- For $j=5$ (Smartwatch): $v_5 = 250$, $w_5 = 0.3$
- For $j=6$ (Tablet): $v_6 = 600$, $w_6 = 1.5$
- For $j=7$ (Bluetooth Speaker): $v_7 = 150$, $w_7 = 1.0$
- For $j=8$ (Keyboard): $v_8 = 80$, $w_8 = 0.8$
- For $j=9$ (Mouse): $v_9 = 50$, $w_9 = 0.2$
- For $j=10$ (Monitor): $v_{10} = 300$, $w_{10} = 3.0$
- For $j=11$ (Printer): $v_{11} = 400$, $w_{11} = 4.0$
- For $j=12$ (External Hard Drive): $v_{12} = 120$, $w_{12} = 0.5$
- For $j=13$ (Router): $v_{13} = 60$, $w_{13} = 0.3$
- For $j=14$ (Power Bank): $v_{14} = 40$, $w_{14} = 0.4$
- For $j=15$ (Memory Card): $v_{15} = 30$, $w_{15} = 0.05$
- For $j=16$ (USB Flash Drive): $v_{16} = 25$, $w_{16} = 0.02$
- For $j=17$ (Smart Home Hub): $v_{17} = 100$, $w_{17} = 0.6$
- For $j=18$ (Gaming Console): $v_{18} = 500$, $w_{18} = 4.0$
- For $j=19$ (Fitness Tracker): $v_{19} = 90$, $w_{19} = 0.2$
- For $j=20$ (E-Reader): $v_{20} = 180$, $w_{20} = 0.5$

Summary:

Maximize total value of products allocated to shelves, subject to each shelf's total weight not exceeding its capacity, with integer nonnegative allocation variables for each product-shelf pair. All identifiers and coefficients are as above, in the original file order.