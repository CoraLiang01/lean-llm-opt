Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv)

**Parameters:**
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of shelf $i$ (from capacity.csv, column "Capacity", matched by ShelfID)

**Data:**

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
\max \sum_{i=1}^{10} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Shelf capacity constraints:**
   \[
   \sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
   \]

2. **Minimum allocation for the first product ("Smartphone"):**
   \[
   \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\quad \forall j \in \text{Products}
   \]

**Where:**
- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (nonnegative integer)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see table above)

**All data and constraints are as retrieved and required by the user query.**