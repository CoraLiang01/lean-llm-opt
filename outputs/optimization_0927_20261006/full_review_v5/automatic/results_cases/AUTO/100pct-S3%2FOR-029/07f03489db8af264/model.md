Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

Let $S$ be the set of shelves (displays), indexed by ShelfID, and $P$ be the set of products, indexed in the order given by the products.csv file. Let $v_j$ be the value and $w_j$ the weight of product $j$. Let $C_i$ be the capacity of shelf $i$.

#### Sets and Indices

- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv, in source order)
- $j \in \{1,2,\ldots,20\}$ (products in source order from products.csv)

#### Parameters

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

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in S$, $j \in P$

#### Objective

$$
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):

   $$
   \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S
   $$

2. **Minimum Total Quantity of First Product** (Smartphone):

   $$
   \sum_{i \in S} x_{i1} \geq 5
   $$

3. **Nonnegativity and Integrality**:

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   $$

---

**All coefficients and identifiers are as retrieved and in original source order.**