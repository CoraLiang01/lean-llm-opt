Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{\text{1},\text{2},\ldots,\text{10}\}$ (ShelfID from capacity.csv) and $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv).

All $x_{ij}$ are nonnegative integers.

---

**Parameters:**

- Shelf capacities (from capacity.csv):

  | ShelfID | Capacity |
  |---------|----------|
  | 1       | 5        |
  | 2       | 7        |
  | 3       | 6        |
  | 4       | 8        |
  | 5       | 5.5      |
  | 6       | 9        |
  | 7       | 6.5      |
  | 8       | 7.5      |
  | 9       | 8.2      |
  | 10      | 5.7      |

- Product values and weights (from products.csv):

  | ProductName            | Value | Weight |
  |------------------------|-------|--------|
  | Smartphone             | 200   | 1      |
  | Laptop                 | 1500  | 5      |
  | Headphones             | 100   | 0.5    |
  | Camera                 | 800   | 2      |
  | Smartwatch             | 250   | 0.3    |
  | Tablet                 | 600   | 1.5    |
  | Bluetooth Speaker      | 150   | 1      |
  | Keyboard               | 80    | 0.8    |
  | Mouse                  | 50    | 0.2    |
  | Monitor                | 300   | 3      |
  | Printer                | 400   | 4      |
  | External Hard Drive    | 120   | 0.5    |
  | Router                 | 60    | 0.3    |
  | Power Bank             | 40    | 0.4    |
  | Memory Card            | 30    | 0.05   |
  | USB Flash Drive        | 25    | 0.02   |
  | Smart Home Hub         | 100   | 0.6    |
  | Gaming Console         | 500   | 4      |
  | Fitness Tracker        | 90    | 0.2    |
  | E-Reader               | 180   | 0.5    |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$ as given above.

**Subject to:**

1. **Shelf capacity constraints (for each shelf $i$):**
   \[
   \sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
   \]
   where $w_j$ is the weight of product $j$ and $c_i$ is the capacity of shelf $i$.

2. **Minimum total quantity of the first product ("Smartphone") across all shelves:**
   \[
   \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}
   \]

---

**Explicitly, the indices and coefficients are:**

- $i$ (ShelfID): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- $j$ (ProductName): Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader
- $v_j$ (Value): as in the table above
- $w_j$ (Weight): as in the table above
- $c_i$ (Capacity): as in the table above

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \Big[200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} \\
&\quad + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} \\
&\quad + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} \\
&\quad + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}} \Big] \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j\,x_{1j} \leq 5 \\
& \sum_{j=1}^{20} w_j\,x_{2j} \leq 7 \\
& \sum_{j=1}^{20} w_j\,x_{3j} \leq 6 \\
& \sum_{j=1}^{20} w_j\,x_{4j} \leq 8 \\
& \sum_{j=1}^{20} w_j\,x_{5j} \leq 5.5 \\
& \sum_{j=1}^{20} w_j\,x_{6j} \leq 9 \\
& \sum_{j=1}^{20} w_j\,x_{7j} \leq 6.5 \\
& \sum_{j=1}^{20} w_j\,x_{8j} \leq 7.5 \\
& \sum_{j=1}^{20} w_j\,x_{9j} \leq 8.2 \\
& \sum_{j=1}^{20} w_j\,x_{10j} \leq 5.7 \\
& \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}
\end{align*}
\]