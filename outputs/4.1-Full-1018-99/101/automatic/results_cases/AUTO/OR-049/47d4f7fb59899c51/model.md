Let $x_{ij}$ denote the number of units of product $j$ placed on shelf $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)
- $j \in \{$
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
$\}$ (ProductName)

**Parameters:**

| ProductName             | Value | Weight |
|-------------------------|-------|--------|
| Smartphone              | 200   | 1.0    |
| Laptop                  | 1500  | 5.0    |
| Headphones              | 100   | 0.5    |
| Camera                  | 800   | 2.0    |
| Smartwatch              | 250   | 0.3    |
| Tablet                  | 600   | 1.5    |
| Bluetooth Speaker       | 150   | 1.0    |
| Keyboard                | 80    | 0.8    |
| Mouse                   | 50    | 0.2    |
| Monitor                 | 300   | 3.0    |
| Printer                 | 400   | 4.0    |
| External Hard Drive     | 120   | 0.5    |
| Router                  | 60    | 0.3    |
| Power Bank              | 40    | 0.4    |
| Memory Card             | 30    | 0.05   |
| USB Flash Drive         | 25    | 0.02   |
| Smart Home Hub          | 100   | 0.6    |
| Gaming Console          | 500   | 4.0    |
| Fitness Tracker         | 90    | 0.2    |
| E-Reader                | 180   | 0.5    |

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

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

**Subject to:**

For each shelf $i$:
\[
\sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of shelf $i$.

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Explicitly:**

Let Products = {Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader}

Let $v_j$ and $w_j$ be as in the table above, and $C_i$ as in the shelf table above.

**Decision variables:** $x_{ij}$ = number of units of product $j$ on shelf $i$, integer, $\geq 0$.

**Objective:**
\[
\max \sum_{i=1}^{10} \Big(
200\,x_{i,\text{Smartphone}} +
1500\,x_{i,\text{Laptop}} +
100\,x_{i,\text{Headphones}} +
800\,x_{i,\text{Camera}} +
250\,x_{i,\text{Smartwatch}} +
600\,x_{i,\text{Tablet}} +
150\,x_{i,\text{Bluetooth Speaker}} +
80\,x_{i,\text{Keyboard}} +
50\,x_{i,\text{Mouse}} +
300\,x_{i,\text{Monitor}} +
400\,x_{i,\text{Printer}} +
120\,x_{i,\text{External Hard Drive}} +
60\,x_{i,\text{Router}} +
40\,x_{i,\text{Power Bank}} +
30\,x_{i,\text{Memory Card}} +
25\,x_{i,\text{USB Flash Drive}} +
100\,x_{i,\text{Smart Home Hub}} +
500\,x_{i,\text{Gaming Console}} +
90\,x_{i,\text{Fitness Tracker}} +
180\,x_{i,\text{E-Reader}}
\Big)
\]

**Constraints:**

For each shelf $i$:

- Shelf 1:
  \[
  1.0\,x_{1,\text{Smartphone}} + 5.0\,x_{1,\text{Laptop}} + 0.5\,x_{1,\text{Headphones}} + 2.0\,x_{1,\text{Camera}} + 0.3\,x_{1,\text{Smartwatch}} + 1.5\,x_{1,\text{Tablet}} + 1.0\,x_{1,\text{Bluetooth Speaker}} + 0.8\,x_{1,\text{Keyboard}} + 0.2\,x_{1,\text{Mouse}} + 3.0\,x_{1,\text{Monitor}} + 4.0\,x_{1,\text{Printer}} + 0.5\,x_{1,\text{External Hard Drive}} + 0.3\,x_{1,\text{Router}} + 0.4\,x_{1,\text{Power Bank}} + 0.05\,x_{1,\text{Memory Card}} + 0.02\,x_{1,\text{USB Flash Drive}} + 0.6\,x_{1,\text{Smart Home Hub}} + 4.0\,x_{1,\text{Gaming Console}} + 0.2\,x_{1,\text{Fitness Tracker}} + 0.5\,x_{1,\text{E-Reader}} \leq 5.0
  \]
- Shelf 2:
  \[
  \ldots \leq 7.0
  \]
- Shelf 3:
  \[
  \ldots \leq 6.0
  \]
- Shelf 4:
  \[
  \ldots \leq 8.0
  \]
- Shelf 5:
  \[
  \ldots \leq 5.5
  \]
- Shelf 6:
  \[
  \ldots \leq 9.0
  \]
- Shelf 7:
  \[
  \ldots \leq 6.5
  \]
- Shelf 8:
  \[
  \ldots \leq 7.5
  \]
- Shelf 9:
  \[
  \ldots \leq 8.2
  \]
- Shelf 10:
  \[
  \ldots \leq 5.7
  \]

**And for all $i, j$:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]