Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Sets and Indices:**
- $i \in \{\text{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader}\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of shelf $i$ (from "Capacity" in capacity.csv)
- $v_j$ = Value of product $j$ (from "Value" in products.csv)
- $w_j$ = Weight of product $j$ (from "Weight" in products.csv)

**Data:**

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

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all products}\}} v_j x_{ij}
$$

Subject to:

1. **Shelf Capacity Constraints** (for each shelf $i$):
$$
\sum_{j} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

2. **Minimum Quantity of First Product (Smartphone) Across All Displays:**
$$
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Where:**

- $c_1 = 5$, $c_2 = 7$, $c_3 = 6$, $c_4 = 8$, $c_5 = 5.5$, $c_6 = 9$, $c_7 = 6.5$, $c_8 = 7.5$, $c_9 = 8.2$, $c_{10} = 5.7$
- $v_j$ and $w_j$ as listed above for each product $j$ in source order.

**Decision variables:**
- $x_{ij}$: number of units of product $j$ placed on shelf $i$, integer and $\geq 0$.