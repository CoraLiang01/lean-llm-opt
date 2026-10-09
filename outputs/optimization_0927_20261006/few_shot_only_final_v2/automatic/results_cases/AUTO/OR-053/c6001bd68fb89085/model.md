**Sets:**
- $i \in \{1,2,\ldots,10\}$: ShelfID (from capacity.csv, ShelfID 1–10, in order)
- $j \in \{1,2,\ldots,20\}$: ProductName (from products.csv, ProductName 1–20, in order)

**Parameters:**
- $C_i$: Capacity of shelf $i$ (from capacity.csv)
- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$

**Mathematical Model:**

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

**Parameter Values (from source, in order):**

- Shelf capacities ($C_i$):

| $i$ (ShelfID) | $C_i$ (Capacity) |
|:-------------:|:----------------:|
| 1             | 500              |
| 2             | 700              |
| 3             | 600              |
| 4             | 800              |
| 5             | 550              |
| 6             | 900              |
| 7             | 650              |
| 8             | 750              |
| 9             | 820              |
| 10            | 570              |

- Product values ($v_j$) and weights ($w_j$):

| $j$ (ProductName) | $v_j$ (Value) | $w_j$ (Weight) |
|:-----------------:|:-------------:|:--------------:|
| 1                 | 50            | 10             |
| 2                 | 70            | 20             |
| 3                 | 30            | 5              |
| 4                 | 60            | 15             |
| 5                 | 80            | 25             |
| 6                 | 90            | 30             |
| 7                 | 40            | 12             |
| 8                 | 100           | 35             |
| 9                 | 55            | 10             |
| 10                | 75            | 20             |
| 11                | 65            | 18             |
| 12                | 95            | 28             |
| 13                | 45            | 8              |
| 14                | 85            | 22             |
| 15                | 70            | 25             |
| 16                | 110           | 40             |
| 17                | 50            | 14             |
| 18                | 60            | 16             |
| 19                | 120           | 50             |
| 20                | 100           | 30             |

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
\]

Where $C_i$, $v_j$, and $w_j$ are as listed above.