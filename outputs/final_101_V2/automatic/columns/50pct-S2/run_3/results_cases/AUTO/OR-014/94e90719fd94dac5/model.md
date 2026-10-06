Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

Parameters:

- Shelves (from capacity.csv, in order):

    1. ShelfID: 1, Capacity: 5.0  
    2. ShelfID: 2, Capacity: 7.0  
    3. ShelfID: 3, Capacity: 6.0  
    4. ShelfID: 4, Capacity: 8.0  
    5. ShelfID: 5, Capacity: 5.5  
    6. ShelfID: 6, Capacity: 9.0  
    7. ShelfID: 7, Capacity: 6.5  
    8. ShelfID: 8, Capacity: 7.5  
    9. ShelfID: 9, Capacity: 8.2  
    10. ShelfID: 10, Capacity: 5.7  

- Products (from products.csv, in order):

    1. Smartphone, Value: 200, Weight: 1.0  
    2. Laptop, Value: 1500, Weight: 5.0  
    3. Headphones, Value: 100, Weight: 0.5  
    4. Camera, Value: 800, Weight: 2.0  
    5. Smartwatch, Value: 250, Weight: 0.3  
    6. Tablet, Value: 600, Weight: 1.5  
    7. Bluetooth Speaker, Value: 150, Weight: 1.0  
    8. Keyboard, Value: 80, Weight: 0.8  
    9. Mouse, Value: 50, Weight: 0.2  
    10. Monitor, Value: 300, Weight: 3.0  
    11. Printer, Value: 400, Weight: 4.0  
    12. External Hard Drive, Value: 120, Weight: 0.5  
    13. Router, Value: 60, Weight: 0.3  
    14. Power Bank, Value: 40, Weight: 0.4  
    15. Memory Card, Value: 30, Weight: 0.05  
    16. USB Flash Drive, Value: 25, Weight: 0.02  
    17. Smart Home Hub, Value: 100, Weight: 0.6  
    18. Gaming Console, Value: 500, Weight: 4.0  
    19. Fitness Tracker, Value: 90, Weight: 0.2  
    20. E-Reader, Value: 180, Weight: 0.5  

Model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to, for each shelf $i$ (ShelfID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of shelf $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Explicitly, for each shelf $i$ (ShelfID from 1 to 10):

$$
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

Where the products and their weights are:

- Smartphone: 1.0
- Laptop: 5.0
- Headphones: 0.5
- Camera: 2.0
- Smartwatch: 0.3
- Tablet: 1.5
- Bluetooth Speaker: 1.0
- Keyboard: 0.8
- Mouse: 0.2
- Monitor: 3.0
- Printer: 4.0
- External Hard Drive: 0.5
- Router: 0.3
- Power Bank: 0.4
- Memory Card: 0.05
- USB Flash Drive: 0.02
- Smart Home Hub: 0.6
- Gaming Console: 4.0
- Fitness Tracker: 0.2
- E-Reader: 0.5

And the shelf capacities are:

- Shelf 1: 5.0
- Shelf 2: 7.0
- Shelf 3: 6.0
- Shelf 4: 8.0
- Shelf 5: 5.5
- Shelf 6: 9.0
- Shelf 7: 6.5
- Shelf 8: 7.5
- Shelf 9: 8.2
- Shelf 10: 5.7

All $x_{ij}$ are nonnegative integers.