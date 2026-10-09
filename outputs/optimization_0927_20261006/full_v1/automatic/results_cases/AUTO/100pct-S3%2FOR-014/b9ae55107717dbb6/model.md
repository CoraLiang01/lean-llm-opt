Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

**Sets and Indices:**
- $i \in \{\text{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader}\}$ (ProductName from products.csv)

**Parameters:**

From capacity.csv (in source order):

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

From products.csv (in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

**Mathematical Model:**

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to (for each shelf $i$):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of shelf $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Explicitly, with all identifiers and coefficients:**

Let $x_{ij}$ be the number of units of product $j$ on shelf $i$.

**Objective:**
\[
\max \Bigg(
\sum_{i=1}^{10} \Big[
200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}}
\Big] \Bigg)
\]

**Constraints:**

For each shelf $i$ (with $C_i$ as below):

- Shelf 1: $C_1 = 5.0$
- Shelf 2: $C_2 = 7.0$
- Shelf 3: $C_3 = 6.0$
- Shelf 4: $C_4 = 8.0$
- Shelf 5: $C_5 = 5.5$
- Shelf 6: $C_6 = 9.0$
- Shelf 7: $C_7 = 6.5$
- Shelf 8: $C_8 = 7.5$
- Shelf 9: $C_9 = 8.2$
- Shelf 10: $C_{10} = 5.7$

For each $i = 1, \ldots, 10$:
\[
\begin{align*}
&1.0\,x_{i,\text{Smartphone}} + 5.0\,x_{i,\text{Laptop}} + 0.5\,x_{i,\text{Headphones}} + 2.0\,x_{i,\text{Camera}} + 0.3\,x_{i,\text{Smartwatch}} + 1.5\,x_{i,\text{Tablet}} + 1.0\,x_{i,\text{Bluetooth Speaker}} + 0.8\,x_{i,\text{Keyboard}} + 0.2\,x_{i,\text{Mouse}} + 3.0\,x_{i,\text{Monitor}} \\
&\quad + 4.0\,x_{i,\text{Printer}} + 0.5\,x_{i,\text{External Hard Drive}} + 0.3\,x_{i,\text{Router}} + 0.4\,x_{i,\text{Power Bank}} + 0.05\,x_{i,\text{Memory Card}} + 0.02\,x_{i,\text{USB Flash Drive}} + 0.6\,x_{i,\text{Smart Home Hub}} + 4.0\,x_{i,\text{Gaming Console}} + 0.2\,x_{i,\text{Fitness Tracker}} + 0.5\,x_{i,\text{E-Reader}} \leq C_i
\end{align*}
\]

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all 20 products above}\}
\]