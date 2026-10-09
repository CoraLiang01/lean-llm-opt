Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$: ShelfID (displays)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$: ProductName (products)

**Parameters:**

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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \Bigg(
    200\, x_{i,\text{Smartphone}} +
    1500\, x_{i,\text{Laptop}} +
    100\, x_{i,\text{Headphones}} +
    800\, x_{i,\text{Camera}} +
    250\, x_{i,\text{Smartwatch}} +
    600\, x_{i,\text{Tablet}} +
    150\, x_{i,\text{Bluetooth Speaker}} +
    80\, x_{i,\text{Keyboard}} +
    50\, x_{i,\text{Mouse}} +
    300\, x_{i,\text{Monitor}} +
    400\, x_{i,\text{Printer}} +
    120\, x_{i,\text{External Hard Drive}} +
    60\, x_{i,\text{Router}} +
    40\, x_{i,\text{Power Bank}} +
    30\, x_{i,\text{Memory Card}} +
    25\, x_{i,\text{USB Flash Drive}} +
    100\, x_{i,\text{Smart Home Hub}} +
    500\, x_{i,\text{Gaming Console}} +
    90\, x_{i,\text{Fitness Tracker}} +
    180\, x_{i,\text{E-Reader}}
\Bigg)
\]

**Subject to:**

For each shelf $i$:

\[
\begin{align*}
&1\, x_{i,\text{Smartphone}} + 5\, x_{i,\text{Laptop}} + 0.5\, x_{i,\text{Headphones}} + 2\, x_{i,\text{Camera}} + 0.3\, x_{i,\text{Smartwatch}} + 1.5\, x_{i,\text{Tablet}} + 1\, x_{i,\text{Bluetooth Speaker}} \\
&\quad + 0.8\, x_{i,\text{Keyboard}} + 0.2\, x_{i,\text{Mouse}} + 3\, x_{i,\text{Monitor}} + 4\, x_{i,\text{Printer}} + 0.5\, x_{i,\text{External Hard Drive}} + 0.3\, x_{i,\text{Router}} \\
&\quad + 0.4\, x_{i,\text{Power Bank}} + 0.05\, x_{i,\text{Memory Card}} + 0.02\, x_{i,\text{USB Flash Drive}} + 0.6\, x_{i,\text{Smart Home Hub}} + 4\, x_{i,\text{Gaming Console}} \\
&\quad + 0.2\, x_{i,\text{Fitness Tracker}} + 0.5\, x_{i,\text{E-Reader}} \leq \text{Capacity}_i
\end{align*}
\]
where $\text{Capacity}_i$ is as follows:
- ShelfID 1: $5$
- ShelfID 2: $7$
- ShelfID 3: $6$
- ShelfID 4: $8$
- ShelfID 5: $5.5$
- ShelfID 6: $9$
- ShelfID 7: $6.5$
- ShelfID 8: $7.5$
- ShelfID 9: $8.2$
- ShelfID 10: $5.7$

**Minimum total quantity for the first product (Smartphone):**
\[
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
\]

**Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all products above}\}
\]