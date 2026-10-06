Let $x_{ij}$ be the number of units of product $j$ placed on display $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv, both in the order given.

**Parameters:**

- Let $S$ be the set of ShelfID (from capacity.csv, in order):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Let $P$ be the set of ProductName (from products.csv, in order):  
  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$
- Let $v_j$ be the Value of product $j$ (from products.csv).
- Let $w_j$ be the Weight of product $j$ (from products.csv).
- Let $C_i$ be the Capacity of display $i$ (from capacity.csv).

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

1. **Display Capacity Constraints:**  
   For each display $i \in S$,
   $$
   \sum_{j \in P} w_j \cdot x_{ij} \leq C_i
   $$

2. **Minimum Quantity of First Product (Smartphone):**
   $$
   \sum_{i \in S} x_{i, \text{Smartphone}} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
   $$

---

**Numerical Data:**

- ShelfID and Capacity (from capacity.csv, in order):

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

- ProductName, Value, and Weight (from products.csv, in order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

---

**Full Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j\, x_{i j} \leq C_i \quad \forall i=1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i,1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
$$

Where $v_j$ and $w_j$ are as listed above, $C_i$ as above, and $x_{ij}$ is the number of units of product $j$ placed on display $i$.