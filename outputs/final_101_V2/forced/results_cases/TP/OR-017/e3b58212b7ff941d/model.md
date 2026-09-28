##### Decision Variables

$x_i \geq 0$ (integer): Number of units of product $i$ (where $i$ is a ‘ZZ’ SKU) to fulfill.

##### Parameters

- $r_i$: Revenue per unit of product $i$.
- $d_i$: Demand for product $i$.
- $s_i$: Initial inventory of product $i$.

Data (in source order):

| SKU    | $r_i$  | $d_i$ | $s_i$  |
|--------|--------|-------|--------|
| ZZ2AO  | 24.38  | 2     | 10.0   |
| ZZDW7  | 30.12  | 4     | 20.0   |
| ZZM1A  | 19.52  | 82    | 530.0  |
| ZZNC5  | 10.79  | 2     | 10.0   |
| ZZX6K  | 111.81 | 2     | 10.0   |

##### Objective Function

$$
\max \; 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
$$

##### Constraints

For each SKU $i$:
- $0 \leq x_i \leq \min\{d_i, s_i\}$

Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq 2 \\
0 \leq x_{\text{ZZDW7}} &\leq 4 \\
0 \leq x_{\text{ZZM1A}} &\leq 82 \\
0 \leq x_{\text{ZZNC5}} &\leq 2 \\
0 \leq x_{\text{ZZX6K}} &\leq 2 \\
\end{align*}
\]

$x_i$ are integer variables.