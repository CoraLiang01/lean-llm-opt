Let $x_i$ be the number of units of product $i$ to fulfill, for each product $i$ in the table below.

**Parameters (from data):**

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|----------------|-----------------|---------------|---------------------------|
| sku_I27        | 238             | 6             | 30                        |
| sku_I499       | 287             | 4             | 20                        |
| sku_I719       | 268             | 16            | 80                        |
| sku_T18        | 318             | 14            | 70                        |
| sku_T29        | 207             | 4             | 20                        |
| sku_T39        | 258             | 32            | 160                       |
| sku_T499       | 249             | 8             | 40                        |
| sku_T9         | 227             | 2             | 10                        |
| sku_3081       | 198             | 10            | 50                        |
| sku_339        | 254             | 8             | 40                        |
| sku_3799       | 246             | 18            | 90                        |
| sku_439        | 258             | 2             | 10                        |
| sku_539        | 268             | 4             | 20                        |
| sku_61399      | 278             | 8             | 40                        |
| sku_628        | 268             | 2             | 10                        |
| sku_708        | 298             | 198           | 990                       |
| sku_77         | 258             | 32            | 160                       |
| sku_79         | 315             | 18            | 90                        |
| sku_799        | 264             | 570           | 2870                      |
| sku_8499       | 238             | 6             | 30                        |
| sku_89         | 258             | 26            | 130                       |
| sku_897        | 268             | 6             | 30                        |
| sku_9699       | 288             | 33            | 170                       |
| sku_bobo       | 228             | 33            | 170                       |

**Decision variables:**

$x_i \in \mathbb{Z}_{\geq 0}$ for each product $i$ (number of units of product $i$ to fulfill)

---

**Objective:**

$$
\max \sum_{i} r_i x_i
$$

where $r_i$ is the Revenue for product $i$.

---

**Constraints:**

For each product $i$:

1. **Demand cannot be exceeded:**
   $$
   x_i \leq d_i
   $$
   where $d_i$ is the Demand for product $i$.

2. **Cannot fulfill more than available inventory:**
   $$
   x_i \leq s_i
   $$
   where $s_i$ is the Initial Inventory for product $i$.

3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}
   $$

---

**Explicitly, for each product $i$ in the table:**

For example, for sku_I27:
- $x_{\text{sku\_I27}} \leq 6$
- $x_{\text{sku\_I27}} \leq 30$
- $x_{\text{sku\_I27}} \in \mathbb{Z}_{\geq 0}$

...and similarly for all other products.

---

**Summary of Model:**

$$
\begin{align*}
\max\ & \sum_{i} r_i x_i \\
\text{s.t.}\quad
& x_i \leq d_i \quad \forall i \\
& x_i \leq s_i \quad \forall i \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

where $i$ indexes the products as listed above, with $r_i$, $d_i$, $s_i$ as given in the table.