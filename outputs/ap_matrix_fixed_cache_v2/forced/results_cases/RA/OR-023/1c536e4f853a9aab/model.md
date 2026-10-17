Let $x_i$ denote the number of units of product $i$ (with Product_Reference as below) to fulfill.

**Parameters (from data):**

| Product_Reference         | Revenue | Demand | Initial Inventory |
|--------------------------|---------|--------|------------------|
| ELE-SMA-10000463         | 4.0     | 295    | 2000.0           |
| ELE-SMA-10000487         | 14.0    | 1002   | 7000.0           |
| ELE-SMA-10003333         | 14.0    | 958    | 7000.0           |
| ELE-SMA-10009012         | 4.0     | 777    | 6000.0           |
| ELE-SMA-10009999         | 4.0     | 271    | 2000.0           |
| ELE-SMA-10011234         | 4.0     | 244    | 2000.0           |
| ELE-SMA-10027456         | 14.0    | 990    | 7000.0           |
| ELE-SMA-10028567         | 14.0    | 1000   | 7000.0           |

**Decision variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ in the table above.

---

**Objective:**

$$
\max \left(
4.0\, x_{\text{ELE-SMA-10000463}}
+ 14.0\, x_{\text{ELE-SMA-10000487}}
+ 14.0\, x_{\text{ELE-SMA-10003333}}
+ 4.0\, x_{\text{ELE-SMA-10009012}}
+ 4.0\, x_{\text{ELE-SMA-10009999}}
+ 4.0\, x_{\text{ELE-SMA-10011234}}
+ 14.0\, x_{\text{ELE-SMA-10027456}}
+ 14.0\, x_{\text{ELE-SMA-10028567}}
\right)
$$

---

**Constraints:**

For each product $i$:

- Demand fulfillment cannot exceed demand:
  $$
  x_i \leq \text{Demand}_i
  $$
- Cannot fulfill more than initial inventory:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Explicitly, for each product:

1. $0 \leq x_{\text{ELE-SMA-10000463}} \leq \min\{295,\ 2000.0\} = 295$
2. $0 \leq x_{\text{ELE-SMA-10000487}} \leq \min\{1002,\ 7000.0\} = 1002$
3. $0 \leq x_{\text{ELE-SMA-10003333}} \leq \min\{958,\ 7000.0\} = 958$
4. $0 \leq x_{\text{ELE-SMA-10009012}} \leq \min\{777,\ 6000.0\} = 777$
5. $0 \leq x_{\text{ELE-SMA-10009999}} \leq \min\{271,\ 2000.0\} = 271$
6. $0 \leq x_{\text{ELE-SMA-10011234}} \leq \min\{244,\ 2000.0\} = 244$
7. $0 \leq x_{\text{ELE-SMA-10027456}} \leq \min\{990,\ 7000.0\} = 990$
8. $0 \leq x_{\text{ELE-SMA-10028567}} \leq \min\{1000,\ 7000.0\} = 1000$

---

**Complete Model:**

$$
\begin{align*}
\max\quad & 4.0\, x_{\text{ELE-SMA-10000463}}
+ 14.0\, x_{\text{ELE-SMA-10000487}}
+ 14.0\, x_{\text{ELE-SMA-10003333}}
+ 4.0\, x_{\text{ELE-SMA-10009012}}
+ 4.0\, x_{\text{ELE-SMA-10009999}}
+ 4.0\, x_{\text{ELE-SMA-10011234}}
+ 14.0\, x_{\text{ELE-SMA-10027456}}
+ 14.0\, x_{\text{ELE-SMA-10028567}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{ELE-SMA-10000463}} \leq 295 \\
& 0 \leq x_{\text{ELE-SMA-10000487}} \leq 1002 \\
& 0 \leq x_{\text{ELE-SMA-10003333}} \leq 958 \\
& 0 \leq x_{\text{ELE-SMA-10009012}} \leq 777 \\
& 0 \leq x_{\text{ELE-SMA-10009999}} \leq 271 \\
& 0 \leq x_{\text{ELE-SMA-10011234}} \leq 244 \\
& 0 \leq x_{\text{ELE-SMA-10027456}} \leq 990 \\
& 0 \leq x_{\text{ELE-SMA-10028567}} \leq 1000 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

Where $i$ indexes the eight products listed above by their Product_Reference.