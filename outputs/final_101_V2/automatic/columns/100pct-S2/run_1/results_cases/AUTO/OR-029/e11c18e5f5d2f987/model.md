Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from the first table and $j$ indexes ProductName from the second table, both in the order given.

**Parameters:**

- Let $S$ be the set of shelves (displays), with ShelfID as below.
- Let $P$ be the set of products, with ProductName as below.
- $c_i$ = Capacity of shelf $i$ (from "capacity.csv")
- $v_j$ = Value of product $j$ (from "products.csv")
- $w_j$ = Weight of product $j$ (from "products.csv")

**Data (in source order):**

Shelves (displays):

| $i$ | ShelfID | Capacity |
|----|---------|----------|
| 1  | 1       | 5        |
| 2  | 2       | 7        |
| 3  | 3       | 6        |
| 4  | 4       | 8        |
| 5  | 5       | 5.5      |
| 6  | 6       | 9        |
| 7  | 7       | 6.5      |
| 8  | 8       | 7.5      |
| 9  | 9       | 8.2      |
| 10 | 10      | 5.7      |

Products:

| $j$ | ProductName             | Value | Weight |
|-----|-------------------------|-------|--------|
| 1   | Smartphone              | 200   | 1      |
| 2   | Laptop                  | 1500  | 5      |
| 3   | Headphones              | 100   | 0.5    |
| 4   | Camera                  | 800   | 2      |
| 5   | Smartwatch              | 250   | 0.3    |
| 6   | Tablet                  | 600   | 1.5    |
| 7   | Bluetooth Speaker       | 150   | 1      |
| 8   | Keyboard                | 80    | 0.8    |
| 9   | Mouse                   | 50    | 0.2    |
| 10  | Monitor                 | 300   | 3      |
| 11  | Printer                 | 400   | 4      |
| 12  | External Hard Drive     | 120   | 0.5    |
| 13  | Router                  | 60    | 0.3    |
| 14  | Power Bank              | 40    | 0.4    |
| 15  | Memory Card             | 30    | 0.05   |
| 16  | USB Flash Drive         | 25    | 0.02   |
| 17  | Smart Home Hub          | 100   | 0.6    |
| 18  | Gaming Console          | 500   | 4      |
| 19  | Fitness Tracker         | 90    | 0.2    |
| 20  | E-Reader                | 180   | 0.5    |

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

**Constraints:**

1. **Shelf Capacity Constraints:** For each shelf $i$,
   $$
   \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
   $$
   where $c_i$ is the Capacity of shelf $i$ as above.

2. **Minimum Allocation for First Product (Smartphone):**
   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   $$

**All identifiers, coefficients, and constraints are as retrieved and in source order.**