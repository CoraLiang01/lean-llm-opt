**Sets and Indices:**

- Let $I$ be the set of products, indexed by $i$ (from the order in products.csv, $i=1,\ldots,20$).
- Let $J$ be the set of displays (shelves), indexed by $j$ (from the order in capacity.csv, $j=1,\ldots,10$).

**Parameters:**

From products.csv (in source order):

| $i$ | ProductName              | $v_i$ (Value) | $w_i$ (Weight) |
|-----|--------------------------|---------------|----------------|
| 1   | Smartphone               | 200           | 1.0            |
| 2   | Laptop                   | 1500          | 5.0            |
| 3   | Headphones               | 100           | 0.5            |
| 4   | Camera                   | 800           | 2.0            |
| 5   | Smartwatch               | 250           | 0.3            |
| 6   | Tablet                   | 600           | 1.5            |
| 7   | Bluetooth Speaker        | 150           | 1.0            |
| 8   | Keyboard                 | 80            | 0.8            |
| 9   | Mouse                    | 50            | 0.2            |
| 10  | Monitor                  | 300           | 3.0            |
| 11  | Printer                  | 400           | 4.0            |
| 12  | External Hard Drive      | 120           | 0.5            |
| 13  | Router                   | 60            | 0.3            |
| 14  | Power Bank               | 40            | 0.4            |
| 15  | Memory Card              | 30            | 0.05           |
| 16  | USB Flash Drive          | 25            | 0.02           |
| 17  | Smart Home Hub           | 100           | 0.6            |
| 18  | Gaming Console           | 500           | 4.0            |
| 19  | Fitness Tracker          | 90            | 0.2            |
| 20  | E-Reader                 | 180           | 0.5            |

From capacity.csv (in source order):

| $j$ | ShelfID | $C_j$ (Capacity) |
|-----|---------|------------------|
| 1   | 1       | 5.0              |
| 2   | 2       | 7.0              |
| 3   | 3       | 6.0              |
| 4   | 4       | 8.0              |
| 5   | 5       | 5.5              |
| 6   | 6       | 9.0              |
| 7   | 7       | 6.5              |
| 8   | 8       | 7.5              |
| 9   | 9       | 8.2              |
| 10  | 10      | 5.7              |

**Decision Variables:**

- $x_{j,i}$: Number of units of product $i$ placed on display (shelf) $j$.
- $x_{j,i} \in \mathbb{Z}_{\geq 0}$ for all $j=1,\ldots,10$, $i=1,\ldots,20$.

**Mathematical Model:**

**Objective:**
\[
\max \sum_{j=1}^{10} \sum_{i=1}^{20} v_i \, x_{j,i}
\]
where $v_i$ is the value of product $i$ as given above.

**Subject to:**

1. **Display (Shelf) Capacity Constraints:**
   For each display $j=1,\ldots,10$:
   \[
   \sum_{i=1}^{20} w_i \, x_{j,i} \leq C_j
   \]
   where $w_i$ is the weight of product $i$ and $C_j$ is the capacity of display $j$ as given above.

2. **Minimum Total Quantity of First Product:**
   \[
   \sum_{j=1}^{10} x_{j,1} \geq 5
   \]
   (where $x_{j,1}$ corresponds to "Smartphone" for all $j$)

3. **Nonnegativity and Integrality:**
   \[
   x_{j,i} \in \mathbb{Z}_{\geq 0} \qquad \forall j=1,\ldots,10,\; i=1,\ldots,20
   \]

---

**Explicit Data Table Reference:**

- $v_i$ and $w_i$ are taken from products.csv in the order above.
- $C_j$ is taken from capacity.csv in the order above.
- $x_{j,i}$ is the number of units of product $i$ placed on display $j$.

---

**Full Model (with explicit indices):**

\[
\begin{align*}
\max \quad & \sum_{j=1}^{10} \sum_{i=1}^{20} v_i \, x_{j,i} \\[2ex]
\text{s.t.} \quad
& \sum_{i=1}^{20} w_i \, x_{j,i} \leq C_j \qquad \forall j=1,\ldots,10 \\[2ex]
& \sum_{j=1}^{10} x_{j,1} \geq 5 \\[2ex]
& x_{j,i} \in \mathbb{Z}_{\geq 0} \qquad \forall j=1,\ldots,10,\; i=1,\ldots,20
\end{align*}
\]

where all coefficients and indices are as listed above.