Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID) and $j \in \{1,2,\ldots,20\}$ (products in the order listed below). All $x_{ij}$ are nonnegative integers.

Let the products, their values, and weights be as follows (in source order):

| $j$ | Product Name            | Value ($v_j$) | Weight ($w_j$) |
|-----|------------------------|--------------|---------------|
| 1   | Smartphone             | 200          | 1             |
| 2   | Laptop                 | 1500         | 5             |
| 3   | Headphones             | 100          | 0.5           |
| 4   | Camera                 | 800          | 2             |
| 5   | Smartwatch             | 250          | 0.3           |
| 6   | Tablet                 | 600          | 1.5           |
| 7   | Bluetooth Speaker      | 150          | 1             |
| 8   | Keyboard               | 80           | 0.8           |
| 9   | Mouse                  | 50           | 0.2           |
| 10  | Monitor                | 300          | 3             |
| 11  | Printer                | 400          | 4             |
| 12  | External Hard Drive    | 120          | 0.5           |
| 13  | Router                 | 60           | 0.3           |
| 14  | Power Bank             | 40           | 0.4           |
| 15  | Memory Card            | 30           | 0.05          |
| 16  | USB Flash Drive        | 25           | 0.02          |
| 17  | Smart Home Hub         | 100          | 0.6           |
| 18  | Gaming Console         | 500          | 4             |
| 19  | Fitness Tracker        | 90           | 0.2           |
| 20  | E-Reader               | 180          | 0.5           |

Let the shelves and their capacities be as follows (in source order):

| $i$ | ShelfID | Capacity ($C_i$) |
|-----|---------|------------------|
| 1   | 1       | 5                |
| 2   | 2       | 7                |
| 3   | 3       | 6                |
| 4   | 4       | 8                |
| 5   | 5       | 5.5              |
| 6   | 6       | 9                |
| 7   | 7       | 6.5              |
| 8   | 8       | 7.5              |
| 9   | 9       | 8.2              |
| 10  | 10      | 5.7              |

The mathematical model is:

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
\]

**Subject to:**

1. **Shelf Capacity Constraints:** For each shelf $i$,
\[
\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

2. **Minimum Allocation of First Product (Smartphone):**
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. **Nonnegativity and Integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

**Where:**

- $v_j$ and $w_j$ are as listed above for each product $j$.
- $C_i$ is as listed above for each shelf $i$.
- $x_{ij}$ is the number of units of product $j$ placed on shelf $i$.

All data and constraints are included as specified in the user query and the retrieved files.