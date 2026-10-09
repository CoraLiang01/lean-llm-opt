Let $x_{ij}$ be the number of units of product $j$ (ProductName) placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

  1. ShelfID = 1, Capacity = 5.0  
  2. ShelfID = 2, Capacity = 7.0  
  3. ShelfID = 3, Capacity = 6.0  
  4. ShelfID = 4, Capacity = 8.0  
  5. ShelfID = 5, Capacity = 5.5  
  6. ShelfID = 6, Capacity = 9.0  
  7. ShelfID = 7, Capacity = 6.5  
  8. ShelfID = 8, Capacity = 7.5  
  9. ShelfID = 9, Capacity = 8.2  
  10. ShelfID = 10, Capacity = 5.7  

- Products (from products.csv, in order):

  1. ProductName = Smartphone, Value = 200, Weight = 1.0  
  2. ProductName = Laptop, Value = 1500, Weight = 5.0  
  3. ProductName = Headphones, Value = 100, Weight = 0.5  
  4. ProductName = Camera, Value = 800, Weight = 2.0  
  5. ProductName = Smartwatch, Value = 250, Weight = 0.3  
  6. ProductName = Tablet, Value = 600, Weight = 1.5  
  7. ProductName = Bluetooth Speaker, Value = 150, Weight = 1.0  
  8. ProductName = Keyboard, Value = 80, Weight = 0.8  
  9. ProductName = Mouse, Value = 50, Weight = 0.2  
  10. ProductName = Monitor, Value = 300, Weight = 3.0  
  11. ProductName = Printer, Value = 400, Weight = 4.0  
  12. ProductName = External Hard Drive, Value = 120, Weight = 0.5  
  13. ProductName = Router, Value = 60, Weight = 0.3  
  14. ProductName = Power Bank, Value = 40, Weight = 0.4  
  15. ProductName = Memory Card, Value = 30, Weight = 0.05  
  16. ProductName = USB Flash Drive, Value = 25, Weight = 0.02  
  17. ProductName = Smart Home Hub, Value = 100, Weight = 0.6  
  18. ProductName = Gaming Console, Value = 500, Weight = 4.0  
  19. ProductName = Fitness Tracker, Value = 90, Weight = 0.2  
  20. ProductName = E-Reader, Value = 180, Weight = 0.5  

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

where $x_{ij}$ is the number of units of product $j$ placed on shelf $i$.

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ (see above).

---

**Constraints:**

For each shelf $i$ (ShelfID as above):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of shelf $i$.

---

**Explicitly, for each shelf:**

- ShelfID = 1, Capacity = 5.0:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{1j} \leq 5.0
  $$
- ShelfID = 2, Capacity = 7.0:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{2j} \leq 7.0
  $$
- ShelfID = 3, Capacity = 6.0:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{3j} \leq 6.0
  $$
- ShelfID = 4, Capacity = 8.0:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{4j} \leq 8.0
  $$
- ShelfID = 5, Capacity = 5.5:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{5j} \leq 5.5
  $$
- ShelfID = 6, Capacity = 9.0:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{6j} \leq 9.0
  $$
- ShelfID = 7, Capacity = 6.5:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{7j} \leq 6.5
  $$
- ShelfID = 8, Capacity = 7.5:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{8j} \leq 7.5
  $$
- ShelfID = 9, Capacity = 8.2:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{9j} \leq 8.2
  $$
- ShelfID = 10, Capacity = 5.7:
  $$
  \sum_{j=1}^{20} w_j \cdot x_{10j} \leq 5.7
  $$

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Where:**

- $v_j$ and $w_j$ are the Value and Weight of product $j$ (ProductName) as listed above, in the original order.
- $C_i$ is the Capacity of shelf $i$ (ShelfID) as listed above, in the original order.

---

**All data and identifiers are preserved in original file and row order.**