**Mathematical Model**

Let  
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$  
where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Let  
- $v_j$ = value of product $j$ (from products.csv)  
- $w_j$ = weight of product $j$ (from products.csv)  
- $C_i$ = capacity of shelf $i$ (from capacity.csv)

**Objective:**  
Maximize total value of products on all shelves:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:**  
For each shelf $i$,
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
$$

2. **Non-negativity and Integrality:**  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
$$

---

**Parameter Values (from CSVs):**

- **Shelves and Capacities (capacity.csv):**

| ShelfID | $C_i$ (Capacity) |
|---------|------------------|
| 1       | 750              |
| 2       | 820              |
| 3       | 570              |
| 4       | 800              |
| 5       | 550              |
| 6       | 900              |
| 7       | 650              |
| 8       | 800              |
| 9       | 850              |
| 10      | 900              |

- **Products, Values, and Weights (products.csv):**

| ProductName ($j$) | $v_j$ (Value) | $w_j$ (Weight) |
|-------------------|---------------|---------------|
| 1                 | 55            | 10            |
| 2                 | 75            | 20            |
| 3                 | 65            | 5             |
| 4                 | 60            | 15            |
| 5                 | 80            | 25            |
| 6                 | 90            | 35            |
| 7                 | 40            | 45            |
| 8                 | 100           | 55            |
| 9                 | 55            | 65            |
| 10                | 75            | 20            |
| 11                | 110           | 18            |
| 12                | 50            | 28            |
| 13                | 60            | 8             |
| 14                | 120           | 28            |
| 15                | 70            | 25            |
| 16                | 110           | 40            |
| 17                | 50            | 55            |
| 18                | 60            | 70            |
| 19                | 120           | 85            |
| 20                | 100           | 100           |

---

**Full Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
$$

where $v_j$, $w_j$, and $C_i$ are as listed above.