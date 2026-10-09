Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

Let $S$ be the set of shelves (from capacity.csv), indexed by $i$ with ShelfID as below.
Let $P$ be the set of products (from products.csv), indexed by $j$ with ProductName as below.

Define:
- $v_j$ = Value of product $j$
- $w_j$ = Weight of product $j$
- $C_i$ = Capacity of shelf $i$

#### Sets and Parameters

Shelves (from capacity.csv, in source order):

| $i$ | ShelfID | Capacity |
|-----|---------|----------|
| 1   | 1       | 5        |
| 2   | 2       | 7        |
| 3   | 3       | 6        |
| 4   | 4       | 8        |
| 5   | 5       | 5.5      |
| 6   | 6       | 9        |
| 7   | 7       | 6.5      |
| 8   | 8       | 7.5      |
| 9   | 9       | 8.2      |
| 10  | 10      | 5.7      |

Products (from products.csv, in source order):

| $j$ | ProductName            | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone            | 200   | 1      |
| 2   | Laptop                | 1500  | 5      |
| 3   | Headphones            | 100   | 0.5    |
| 4   | Camera                | 800   | 2      |
| 5   | Smartwatch            | 250   | 0.3    |
| 6   | Tablet                | 600   | 1.5    |
| 7   | Bluetooth Speaker     | 150   | 1      |
| 8   | Keyboard              | 80    | 0.8    |
| 9   | Mouse                 | 50    | 0.2    |
| 10  | Monitor               | 300   | 3      |
| 11  | Printer               | 400   | 4      |
| 12  | External Hard Drive   | 120   | 0.5    |
| 13  | Router                | 60    | 0.3    |
| 14  | Power Bank            | 40    | 0.4    |
| 15  | Memory Card           | 30    | 0.05   |
| 16  | USB Flash Drive       | 25    | 0.02   |
| 17  | Smart Home Hub        | 100   | 0.6    |
| 18  | Gaming Console        | 500   | 4      |
| 19  | Fitness Tracker       | 90    | 0.2    |
| 20  | E-Reader              | 180   | 0.5    |

#### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in \{1,\ldots,10\}$ (ShelfID), $j \in \{1,\ldots,20\}$ (ProductName).

#### Objective

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):

$$
\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\}
$$

That is, for each shelf, the total weight of all products placed does not exceed its capacity.

2. **Minimum Quantity of First Product Across All Shelves** (Smartphone):

$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality**:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
$$

#### Complete Numerical Formulation

Let $x_{ij}$ = number of units of ProductName $j$ on ShelfID $i$.

**Parameters:**

- $v_j$ (Value): as above
- $w_j$ (Weight): as above
- $C_i$ (Capacity): as above

**Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij} \\[2ex]
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10 \\[2ex]
& \sum_{i=1}^{10} x_{i1} \geq 5 \\[2ex]
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\; j = 1,\ldots,20
\end{align*}
$$

Where the mapping of $i$ and $j$ to ShelfID and ProductName is as listed above, and all coefficients are taken directly from the retrieved data.