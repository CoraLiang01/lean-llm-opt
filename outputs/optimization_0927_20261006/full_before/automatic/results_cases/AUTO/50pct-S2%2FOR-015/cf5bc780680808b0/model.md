Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For shelves (from capacity.csv):

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

- For products (from products.csv):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the resource_requirement of product $j$, and $c_i$ is the resource_capacity of shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Explicit Data:**

- $c_i$ (resource_capacity) for $i=1,\ldots,10$:

    - $c_1 = 500$
    - $c_2 = 700$
    - $c_3 = 600$
    - $c_4 = 800$
    - $c_5 = 550$
    - $c_6 = 900$
    - $c_7 = 650$
    - $c_8 = 750$
    - $c_9 = 820$
    - $c_{10} = 570$

- $v_j$ (item_value) and $w_j$ (resource_requirement) for $j=1,\ldots,20$:

    - $v_1 = 50$, $w_1 = 10$
    - $v_2 = 70$, $w_2 = 20$
    - $v_3 = 30$, $w_3 = 5$
    - $v_4 = 60$, $w_4 = 15$
    - $v_5 = 80$, $w_5 = 25$
    - $v_6 = 90$, $w_6 = 30$
    - $v_7 = 40$, $w_7 = 12$
    - $v_8 = 100$, $w_8 = 35$
    - $v_9 = 55$, $w_9 = 10$
    - $v_{10} = 75$, $w_{10} = 20$
    - $v_{11} = 65$, $w_{11} = 18$
    - $v_{12} = 95$, $w_{12} = 28$
    - $v_{13} = 45$, $w_{13} = 8$
    - $v_{14} = 85$, $w_{14} = 22$
    - $v_{15} = 70$, $w_{15} = 25$
    - $v_{16} = 110$, $w_{16} = 40$
    - $v_{17} = 50$, $w_{17} = 14$
    - $v_{18} = 60$, $w_{18} = 16$
    - $v_{19} = 120$, $w_{19} = 50$
    - $v_{20} = 100$, $w_{20} = 30$

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's weight capacity, with integer allocation variables:

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\ & \sum_{j=1}^{20} w_j x_{ij} \leq c_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i,j
\end{align*}
$$

where all coefficients and identifiers are as listed above.