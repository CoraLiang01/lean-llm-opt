##### Decision Variables

$x_i \geq 0$, integer: Number of units of product $i$ (with SKU as below) to fulfill, for each $i \in I$ (the set of all ‘ZZ’ products).

##### Parameters

Let $I = \{\text{4ZZWJ},\ \text{ZZZTA},\ \text{ZZ2AO},\ \text{ZZSZDW},\ \text{ZZX6K}\}$

For each $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $s_i$: Initial inventory of product $i$
- $d_i$: Demand for product $i$

Parameter values:

| SKU      | $r_i$ (Revenue) | $s_i$ (Initial Inventory) | $d_i$ (Demand) |
|----------|-----------------|--------------------------|----------------|
| 4ZZWJ    | 8.56            | 10.0                     | 2              |
| ZZZTA    | 1.58            | 10.0                     | 2              |
| ZZ2AO    | 24.38           | 10.0                     | 2              |
| ZZSZDW   | 110.7           | 30.0                     | 5              |
| ZZX6K    | 111.81          | 10.0                     | 2              |

##### Objective Function

$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

1. Inventory constraint: $x_i \leq s_i,\quad \forall i \in I$
2. Demand constraint: $x_i \leq d_i,\quad \forall i \in I$
3. Nonnegativity and integrality: $x_i \geq 0$ and integer, $\forall i \in I$

##### Complete Model

$$
\begin{align*}
\max\quad & 8.56\,x_{\text{4ZZWJ}} + 1.58\,x_{\text{ZZZTA}} + 24.38\,x_{\text{ZZ2AO}} + 110.7\,x_{\text{ZZSZDW}} + 111.81\,x_{\text{ZZX6K}} \\
\text{s.t.}\quad
& x_{\text{4ZZWJ}} \leq 10.0 \\
& x_{\text{4ZZWJ}} \leq 2 \\
& x_{\text{ZZZTA}} \leq 10.0 \\
& x_{\text{ZZZTA}} \leq 2 \\
& x_{\text{ZZ2AO}} \leq 10.0 \\
& x_{\text{ZZ2AO}} \leq 2 \\
& x_{\text{ZZSZDW}} \leq 30.0 \\
& x_{\text{ZZSZDW}} \leq 5 \\
& x_{\text{ZZX6K}} \leq 10.0 \\
& x_{\text{ZZX6K}} \leq 2 \\
& x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\end{align*}
$$

Where $I = \{\text{4ZZWJ},\ \text{ZZZTA},\ \text{ZZ2AO},\ \text{ZZSZDW},\ \text{ZZX6K}\}$, and parameter values are as listed above.