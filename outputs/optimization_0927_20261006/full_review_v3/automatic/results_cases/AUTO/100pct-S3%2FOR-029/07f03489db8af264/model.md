Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, for $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$, corresponding to the order of shelves and products as given below.

**Sets and Indices:**

- Shelves (Displays): $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with ShelfID as below.
- Products: $j \in \{1,2,\ldots,20\}$, with ProductName as below.

**Parameters:**

- $c_i$: Capacity of shelf $i$ (from "Capacity" column in capacity.csv)
- $v_j$: Value of product $j$ (from "Value" column in products.csv)
- $w_j$: Weight of product $j$ (from "Weight" column in products.csv)

**Shelf (Display) Data (in order):**

| $i$ | ShelfID | $c_i$ (Capacity) |
|---|---------|------------------|
| 1 | 1       | 5                |
| 2 | 2       | 7                |
| 3 | 3       | 6                |
| 4 | 4       | 8                |
| 5 | 5       | 5.5              |
| 6 | 6       | 9                |
| 7 | 7       | 6.5              |
| 8 | 8       | 7.5              |
| 9 | 9       | 8.2              |
|10 | 10      | 5.7              |

**Product Data (in order):**

| $j$ | ProductName             | $v_j$ (Value) | $w_j$ (Weight) |
|---|--------------------------|---------------|----------------|
| 1 | Smartphone               | 200           | 1              |
| 2 | Laptop                   | 1500          | 5              |
| 3 | Headphones               | 100           | 0.5            |
| 4 | Camera                   | 800           | 2              |
| 5 | Smartwatch               | 250           | 0.3            |
| 6 | Tablet                   | 600           | 1.5            |
| 7 | Bluetooth Speaker        | 150           | 1              |
| 8 | Keyboard                 | 80            | 0.8            |
| 9 | Mouse                    | 50            | 0.2            |
|10 | Monitor                  | 300           | 3              |
|11 | Printer                  | 400           | 4              |
|12 | External Hard Drive      | 120           | 0.5            |
|13 | Router                   | 60            | 0.3            |
|14 | Power Bank               | 40            | 0.4            |
|15 | Memory Card              | 30            | 0.05           |
|16 | USB Flash Drive          | 25            | 0.02           |
|17 | Smart Home Hub           | 100           | 0.6            |
|18 | Gaming Console           | 500           | 4              |
|19 | Fitness Tracker          | 90            | 0.2            |
|20 | E-Reader                 | 180           | 0.5            |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:**

For each shelf $i$:
$$
\sum_{j=1}^{20} w_j\, x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

2. **Minimum Placement of First Product:**

$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**All coefficients and identifiers are as retrieved and in original order.**