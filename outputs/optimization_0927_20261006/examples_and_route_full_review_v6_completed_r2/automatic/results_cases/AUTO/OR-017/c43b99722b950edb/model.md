Let $x_i$ be the number of units of product $i$ (SKU) in category 'ZZ' that the store plans to fulfill.

**Parameters (from data):**

| SKU    | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|--------|-----------------|---------------|---------------------------|
| ZZ2AO  | 24.38           | 2             | 10.0                      |
| ZZDW7  | 30.12           | 4             | 20.0                      |
| ZZM1A  | 19.52           | 82            | 530.0                     |
| ZZNC5  | 10.79           | 2             | 10.0                      |
| ZZX6K  | 111.81          | 2             | 10.0                      |

**Decision variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for each SKU $i$ in {ZZ2AO, ZZDW7, ZZM1A, ZZNC5, ZZX6K}

---

**Objective:**

$$
\max \; 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
$$

---

**Constraints:**

For each SKU $i$:

1. Cannot fulfill more than demand:
   $$
   x_i \leq d_i
   $$
   Specifically:
   \begin{align*}
   x_{\text{ZZ2AO}} &\leq 2 \\
   x_{\text{ZZDW7}} &\leq 4 \\
   x_{\text{ZZM1A}} &\leq 82 \\
   x_{\text{ZZNC5}} &\leq 2 \\
   x_{\text{ZZX6K}} &\leq 2 \\
   \end{align*}

2. Cannot fulfill more than initial inventory:
   $$
   x_i \leq s_i
   $$
   Specifically:
   \begin{align*}
   x_{\text{ZZ2AO}} &\leq 10.0 \\
   x_{\text{ZZDW7}} &\leq 20.0 \\
   x_{\text{ZZM1A}} &\leq 530.0 \\
   x_{\text{ZZNC5}} &\leq 10.0 \\
   x_{\text{ZZX6K}} &\leq 10.0 \\
   \end{align*}

3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}
   $$

---

**Complete Model:**

$$
\begin{align*}
\max \quad & 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}} \\
\text{s.t.} \quad
& x_{\text{ZZ2AO}} \leq 2 \\
& x_{\text{ZZ2AO}} \leq 10.0 \\
& x_{\text{ZZDW7}} \leq 4 \\
& x_{\text{ZZDW7}} \leq 20.0 \\
& x_{\text{ZZM1A}} \leq 82 \\
& x_{\text{ZZM1A}} \leq 530.0 \\
& x_{\text{ZZNC5}} \leq 2 \\
& x_{\text{ZZNC5}} \leq 10.0 \\
& x_{\text{ZZX6K}} \leq 2 \\
& x_{\text{ZZX6K}} \leq 10.0 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{ZZ2AO}, \text{ZZDW7}, \text{ZZM1A}, \text{ZZNC5}, \text{ZZX6K}\}
\end{align*}
$$