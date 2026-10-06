Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Parameters:**

- Let $S$ be the set of shelves (displays), with ShelfID from capacity.csv:
  $S = \{1,2,3,4,5,6,7,8,9,10\}$
- Let $P$ be the set of products, with ProductName from products.csv, in order:
  $P = \{$
    Smartphone,
    Laptop,
    Headphones,
    Camera,
    Smartwatch,
    Tablet,
    Bluetooth Speaker,
    Keyboard,
    Mouse,
    Monitor,
    Printer,
    External Hard Drive,
    Router,
    Power Bank,
    Memory Card,
    USB Flash Drive,
    Smart Home Hub,
    Gaming Console,
    Fitness Tracker,
    E-Reader
  $\}$
- Let $v_j$ be the Value of product $j$ (from products.csv).
- Let $w_j$ be the Weight of product $j$ (from products.csv).
- Let $C_i$ be the Capacity of shelf $i$ (from capacity.csv).

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Shelf capacity constraints:** For each shelf $i \in S$,
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

2. **Minimum allocation of first product (Smartphone):**
\[
\sum_{i \in S} x_{i, \text{Smartphone}} \geq 5
\]

3. **Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

---

**Numerical Data:**

- Shelf capacities ($C_i$):

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

- Product values ($v_j$) and weights ($w_j$):

| ProductName             | Value | Weight |
|-------------------------|-------|--------|
| Smartphone              | 200   | 1      |
| Laptop                  | 1500  | 5      |
| Headphones              | 100   | 0.5    |
| Camera                  | 800   | 2      |
| Smartwatch              | 250   | 0.3    |
| Tablet                  | 600   | 1.5    |
| Bluetooth Speaker       | 150   | 1      |
| Keyboard                | 80    | 0.8    |
| Mouse                   | 50    | 0.2    |
| Monitor                 | 300   | 3      |
| Printer                 | 400   | 4      |
| External Hard Drive     | 120   | 0.5    |
| Router                  | 60    | 0.3    |
| Power Bank              | 40    | 0.4    |
| Memory Card             | 30    | 0.05   |
| USB Flash Drive         | 25    | 0.02   |
| Smart Home Hub          | 100   | 0.6    |
| Gaming Console          | 500   | 4      |
| Fitness Tracker         | 90    | 0.2    |
| E-Reader                | 180   | 0.5    |

---

**Complete Model:**

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big[ 200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} \\
&\qquad + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} \\
&\qquad + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} \\
&\qquad + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} \\
&\qquad + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}} \Big] \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j\,x_{1j} \leq 5 \\
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
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
\]

Where $w_j$ is the weight of product $j$ as listed above, and $x_{i,\text{ProductName}}$ is the number of units of that product on shelf $i$.