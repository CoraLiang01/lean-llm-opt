Let $x_{ij}$ be the number of units of product $j$ (with ProductName as below) to be placed on shelf $i$ (with ShelfID as below). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from "capacity.csv", in order):

  | ShelfID |
  |---------|
  | 1       |
  | 2       |
  | 3       |
  | 4       |
  | 5       |
  | 6       |
  | 7       |
  | 8       |
  | 9       |
  | 10      |

  Shelf capacities $C_i$:

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

- Products (from "products.csv", in order):

  | ProductName             | Value | Weight |
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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all ProductName}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to, for each shelf $i$ (with ShelfID as above):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of shelf $i$.

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Explicitly, with all identifiers and coefficients:**

Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$.

- For $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID)
- For $j$ in the following ProductName list:

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

Objective:
\[
\max \sum_{i=1}^{10} \Big(
200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}}
\Big)
\]

For each shelf $i$ (ShelfID as below), the capacity constraint:

For ShelfID 1 (Capacity 5.0):
\[
1.0\,x_{1,\text{Smartphone}} + 5.0\,x_{1,\text{Laptop}} + 0.5\,x_{1,\text{Headphones}} + 2.0\,x_{1,\text{Camera}} + 0.3\,x_{1,\text{Smartwatch}} + 1.5\,x_{1,\text{Tablet}} + 1.0\,x_{1,\text{Bluetooth Speaker}} + 0.8\,x_{1,\text{Keyboard}} + 0.2\,x_{1,\text{Mouse}} + 3.0\,x_{1,\text{Monitor}} + 4.0\,x_{1,\text{Printer}} + 0.5\,x_{1,\text{External Hard Drive}} + 0.3\,x_{1,\text{Router}} + 0.4\,x_{1,\text{Power Bank}} + 0.05\,x_{1,\text{Memory Card}} + 0.02\,x_{1,\text{USB Flash Drive}} + 0.6\,x_{1,\text{Smart Home Hub}} + 4.0\,x_{1,\text{Gaming Console}} + 0.2\,x_{1,\text{Fitness Tracker}} + 0.5\,x_{1,\text{E-Reader}} \leq 5.0
\]

Repeat the same structure for ShelfID 2 through 10, using their respective capacities.

For all $i \in \{1,\ldots,10\}$, $j \in \{\text{all ProductName}\}$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]