Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (from capacity.csv, ShelfID)

- Products ($j$):  
  Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader (from products.csv, ProductName)

- Product values ($v_j$):  
  Smartphone: 200  
  Laptop: 1500  
  Headphones: 100  
  Camera: 800  
  Smartwatch: 250  
  Tablet: 600  
  Bluetooth Speaker: 150  
  Keyboard: 80  
  Mouse: 50  
  Monitor: 300  
  Printer: 400  
  External Hard Drive: 120  
  Router: 60  
  Power Bank: 40  
  Memory Card: 30  
  USB Flash Drive: 25  
  Smart Home Hub: 100  
  Gaming Console: 500  
  Fitness Tracker: 90  
  E-Reader: 180  

- Product weights ($w_j$):  
  Smartphone: 1.0  
  Laptop: 5.0  
  Headphones: 0.5  
  Camera: 2.0  
  Smartwatch: 0.3  
  Tablet: 1.5  
  Bluetooth Speaker: 1.0  
  Keyboard: 0.8  
  Mouse: 0.2  
  Monitor: 3.0  
  Printer: 4.0  
  External Hard Drive: 0.5  
  Router: 0.3  
  Power Bank: 0.4  
  Memory Card: 0.05  
  USB Flash Drive: 0.02  
  Smart Home Hub: 0.6  
  Gaming Console: 4.0  
  Fitness Tracker: 0.2  
  E-Reader: 0.5  

- Shelf capacities ($C_i$):  
  1: 5.0  
  2: 7.0  
  3: 6.0  
  4: 8.0  
  5: 5.5  
  6: 9.0  
  7: 6.5  
  8: 7.5  
  9: 8.2  
  10: 5.7  

---

**Mathematical Model**

**Decision Variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

**Objective:**  
Maximize total value across all shelves:
$$
\max \sum_{i=1}^{10} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (ShelfID from 1 to 10):
$$
\sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i
$$

For all $i$ and $j$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$
- $v_j$ = value of product $j$ (see above)
- $w_j$ = weight of product $j$ (see above)
- $C_i$ = capacity of shelf $i$ (see above)
- $i$ indexes ShelfID from capacity.csv
- $j$ indexes ProductName from products.csv

All data and identifiers are as retrieved and preserved in original order.