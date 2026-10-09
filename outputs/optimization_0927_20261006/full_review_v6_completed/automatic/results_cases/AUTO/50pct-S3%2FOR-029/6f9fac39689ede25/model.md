Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID, in source order)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName, in source order)

**Parameters:**

From capacity.csv (in source order):

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

From products.csv (in source order):

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
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

**Constraints:**

1. **Shelf Capacity Constraints (for each shelf $i$):**
   \[
   \sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
   \]
   where $w_j$ is the Weight of product $j$, $c_i$ is the Capacity of shelf $i$.

2. **Minimum Allocation of First Product (Smartphone):**
   \[
   \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
   \]

**Explicit Data:**

- Shelf capacities $c_i$ (by ShelfID, in order): $[5,\, 7,\, 6,\, 8,\, 5.5,\, 9,\, 6.5,\, 7.5,\, 8.2,\, 5.7]$
- Product values $v_j$ and weights $w_j$ (by ProductName, in order):

| ProductName           | $v_j$ | $w_j$  |
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

**Decision variables:**
- $x_{ij}$: number of units of product $j$ placed on shelf $i$, integer and $\geq 0$.

**Full Model:**

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big(200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} \\
&\qquad + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} \\
&\qquad + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} \\
&\qquad + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}} \Big) \\
\text{s.t.}\quad & \sum_{j} w_j\, x_{1j} \leq 5 \\
& \sum_{j} w_j\, x_{2j} \leq 7 \\
& \sum_{j} w_j\, x_{3j} \leq 6 \\
& \sum_{j} w_j\, x_{4j} \leq 8 \\
& \sum_{j} w_j\, x_{5j} \leq 5.5 \\
& \sum_{j} w_j\, x_{6j} \leq 9 \\
& \sum_{j} w_j\, x_{7j} \leq 6.5 \\
& \sum_{j} w_j\, x_{8j} \leq 7.5 \\
& \sum_{j} w_j\, x_{9j} \leq 8.2 \\
& \sum_{j} w_j\, x_{10j} \leq 5.7 \\
& \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all products}\}
\end{align*}
\]