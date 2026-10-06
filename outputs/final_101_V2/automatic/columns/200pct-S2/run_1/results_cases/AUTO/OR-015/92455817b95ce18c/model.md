Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Define:
- Let $S$ be the set of shelves, indexed by resource_id from capacity.csv.
- Let $P$ be the set of products, indexed by item_name from products.csv.
- For each shelf $i \in S$, let $C_i$ be its resource_capacity from capacity.csv.
- For each product $j \in P$, let $v_j$ be its item_value and $w_j$ its resource_requirement from products.csv.

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i \in S$ (resource_id from capacity.csv):
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in S$, $j \in P$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Parameter Data (in source order):**

capacity.csv

| resource_id | resource_capacity |
|-------------|------------------|
| 1           | 500              |
| 2           | 700              |
| 3           | 600              |
| 4           | 800              |
| 5           | 550              |
| 6           | 900              |
| 7           | 650              |
| 8           | 750              |
| 9           | 820              |
| 10          | 570              |

products.csv

| item_name | item_value | resource_requirement |
|-----------|------------|---------------------|
| 1         | 50         | 10                  |
| 2         | 70         | 20                  |
| 3         | 30         | 5                   |
| 4         | 60         | 15                  |
| 5         | 80         | 25                  |
| 6         | 90         | 30                  |
| 7         | 40         | 12                  |
| 8         | 100        | 35                  |
| 9         | 55         | 10                  |
| 10        | 75         | 20                  |
| 11        | 65         | 18                  |
| 12        | 95         | 28                  |
| 13        | 45         | 8                   |
| 14        | 85         | 22                  |
| 15        | 70         | 25                  |
| 16        | 110        | 40                  |
| 17        | 50         | 14                  |
| 18        | 60         | 16                  |
| 19        | 120        | 50                  |
| 20        | 100        | 30                  |

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{i1j} \leq 500 \\
& \sum_{j=1}^{20} w_j x_{i2j} \leq 700 \\
& \sum_{j=1}^{20} w_j x_{i3j} \leq 600 \\
& \sum_{j=1}^{20} w_j x_{i4j} \leq 800 \\
& \sum_{j=1}^{20} w_j x_{i5j} \leq 550 \\
& \sum_{j=1}^{20} w_j x_{i6j} \leq 900 \\
& \sum_{j=1}^{20} w_j x_{i7j} \leq 650 \\
& \sum_{j=1}^{20} w_j x_{i8j} \leq 750 \\
& \sum_{j=1}^{20} w_j x_{i9j} \leq 820 \\
& \sum_{j=1}^{20} w_j x_{i10j} \leq 570 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
\]

Where for each $j$ (product):

| $j$ | $v_j$ | $w_j$ |
|-----|-------|-------|
| 1   | 50    | 10    |
| 2   | 70    | 20    |
| 3   | 30    | 5     |
| 4   | 60    | 15    |
| 5   | 80    | 25    |
| 6   | 90    | 30    |
| 7   | 40    | 12    |
| 8   | 100   | 35    |
| 9   | 55    | 10    |
| 10  | 75    | 20    |
| 11  | 65    | 18    |
| 12  | 95    | 28    |
| 13  | 45    | 8     |
| 14  | 85    | 22    |
| 15  | 70    | 25    |
| 16  | 110   | 40    |
| 17  | 50    | 14    |
| 18  | 60    | 16    |
| 19  | 120   | 50    |
| 20  | 100   | 30    |