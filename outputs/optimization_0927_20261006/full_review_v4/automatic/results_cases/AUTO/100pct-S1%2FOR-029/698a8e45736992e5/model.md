Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $S$ be the set of shelves (displays), indexed by ShelfID:
$$
S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $P$ be the set of products, indexed in the order returned:
$$
P = \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}
$$

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$ (from products.csv).
Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

#### Parameters

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

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

**Subject to:**

1. **Shelf capacity constraints (for each shelf $i$):**
   \[
   \sum_{j \in P} w_j \, x_{ij} \leq C_i \qquad \forall i \in S
   \]

2. **Minimum total quantity of the first product ("Smartphone") across all shelves:**
   \[
   \sum_{i \in S} x_{i, \text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
   \]

Where:
- $v_j$ and $w_j$ are as given in the table above for each product $j$,
- $C_i$ is as given in the table above for each shelf $i$,
- $x_{ij}$ is the number of units of product $j$ placed on shelf $i$.

All indices and coefficients are as retrieved and in original order.