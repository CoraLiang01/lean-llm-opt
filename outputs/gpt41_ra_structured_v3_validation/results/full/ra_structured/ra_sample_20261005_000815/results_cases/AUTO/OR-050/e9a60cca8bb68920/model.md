Let:
- \( I \) = set of displays (indexed by ShelfID from capacity.csv): \( I = \{1,2,3,4,5,6,7,8,9,10\} \)
- \( J \) = set of products (indexed by ProductName from products.csv, in original order):

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

Parameters:
- \( v_j \) = Value of product \( j \) (from Value column)
- \( w_j \) = Weight of product \( j \) (from Weight column)
- \( C_i \) = Capacity of display \( i \) (from Capacity column)

Decision variables:
- \( x_{ij} \): Number of units of product \( j \) placed on display \( i \)
- Domain: \( x_{ij} \in \mathbb{Z}_{\geq 0} \) (nonnegative integers)

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Display Capacity Constraints** (for each display \( i \)):
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
   \]

2. **Minimum Quantity of First Product Constraint** (Smartphone):
   \[
   \sum_{i \in I} x_{i,1} \geq 5
   \]
   (where \( x_{i,1} \) refers to Smartphone, the first product in products.csv)

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Parameter Table (from CSVs):**

- **Displays (capacity.csv):**

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

- **Products (products.csv):**

| Index | ProductName           | Value | Weight |
|-------|----------------------|-------|--------|
| 1     | Smartphone           | 200   | 1.0    |
| 2     | Laptop               | 1500  | 5.0    |
| 3     | Headphones           | 100   | 0.5    |
| 4     | Camera               | 800   | 2.0    |
| 5     | Smartwatch           | 250   | 0.3    |
| 6     | Tablet               | 600   | 1.5    |
| 7     | Bluetooth Speaker    | 150   | 1.0    |
| 8     | Keyboard             | 80    | 0.8    |
| 9     | Mouse                | 50    | 0.2    |
| 10    | Monitor              | 300   | 3.0    |
| 11    | Printer              | 400   | 4.0    |
| 12    | External Hard Drive  | 120   | 0.5    |
| 13    | Router               | 60    | 0.3    |
| 14    | Power Bank           | 40    | 0.4    |
| 15    | Memory Card          | 30    | 0.05   |
| 16    | USB Flash Drive      | 25    | 0.02   |
| 17    | Smart Home Hub       | 100   | 0.6    |
| 18    | Gaming Console       | 500   | 4.0    |
| 19    | Fitness Tracker      | 90    | 0.2    |
| 20    | E-Reader             | 180   | 0.5    |

---

**Summary:**

- Decision variables: \( x_{ij} \) = number of units of product \( j \) on display \( i \), for all \( i \in \{1,\ldots,10\} \), \( j \in \{1,\ldots,20\} \)
- Objective: maximize total value placed
- Each display's total weight cannot exceed its capacity
- At least 5 units of Smartphone (the first product) must be placed in total
- All variables are nonnegative integers

This is a complete, explicit mathematical optimization model using all provided data and constraints.