##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ (where $i$ is an ‘ELE-S’ product) to fulfill.

##### Parameters

Let $I$ be the set of all ‘ELE-S’ products:

\[
I = \{
\text{ELE-SMA-10000463},
\text{ELE-SMA-10000487},
\text{ELE-SMA-10003333},
\text{ELE-SMA-10009012},
\text{ELE-SMA-10009999},
\text{ELE-SMA-10011234},
\text{ELE-SMA-10027456},
\text{ELE-SMA-10028567}
\}
\]

For each $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $s_i$: Initial inventory of product $i$
- $d_i$: Demand for product $i$

The parameter values are:

| Product ID             | $r_i$ | $s_i$  | $d_i$ |
|------------------------|-------|--------|-------|
| ELE-SMA-10000463       | 4.0   | 2000.0 | 295   |
| ELE-SMA-10000487       | 14.0  | 7000.0 | 1002  |
| ELE-SMA-10003333       | 14.0  | 7000.0 | 958   |
| ELE-SMA-10009012       | 4.0   | 6000.0 | 777   |
| ELE-SMA-10009999       | 4.0   | 2000.0 | 271   |
| ELE-SMA-10011234       | 4.0   | 2000.0 | 244   |
| ELE-SMA-10027456       | 14.0  | 7000.0 | 990   |
| ELE-SMA-10028567       | 14.0  | 7000.0 | 1000  |

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand limits for each product:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I
   \]
   (i.e., cannot fulfill more than available inventory or demand)

2. Integer variables:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & 4.0\, x_{\text{ELE-SMA-10000463}} + 14.0\, x_{\text{ELE-SMA-10000487}} + 14.0\, x_{\text{ELE-SMA-10003333}} + 4.0\, x_{\text{ELE-SMA-10009012}} \\
& + 4.0\, x_{\text{ELE-SMA-10009999}} + 4.0\, x_{\text{ELE-SMA-10011234}} + 14.0\, x_{\text{ELE-SMA-10027456}} + 14.0\, x_{\text{ELE-SMA-10028567}} \\
\text{s.t.} \quad & 0 \leq x_{\text{ELE-SMA-10000463}} \leq 295 \\
& 0 \leq x_{\text{ELE-SMA-10000487}} \leq 1002 \\
& 0 \leq x_{\text{ELE-SMA-10003333}} \leq 958 \\
& 0 \leq x_{\text{ELE-SMA-10009012}} \leq 777 \\
& 0 \leq x_{\text{ELE-SMA-10009999}} \leq 271 \\
& 0 \leq x_{\text{ELE-SMA-10011234}} \leq 244 \\
& 0 \leq x_{\text{ELE-SMA-10027456}} \leq 990 \\
& 0 \leq x_{\text{ELE-SMA-10028567}} \leq 1000 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

Where all parameters are as listed above.