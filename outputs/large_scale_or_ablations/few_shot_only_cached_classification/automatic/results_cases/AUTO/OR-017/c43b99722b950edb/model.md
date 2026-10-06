##### Objective Function

$\quad \max \sum_{i \in \mathcal{ZZ}} \text{Revenue}_i \cdot x_i$

##### Constraints

- Inventory constraints: $\quad x_i \leq \text{Initial Inventory}_i \quad \forall i \in \mathcal{ZZ}$
- Demand constraints: $\quad x_i \leq \text{Demand}_i \quad \forall i \in \mathcal{ZZ}$
- Non-negativity and integrality: $\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{ZZ}$

##### Variable Definitions

- $x_i$: Number of units of product $i$ (classified as ‘ZZ’) to fulfill, integer, $x_i \geq 0$.

##### Retrieved Information

For every product $i$ classified as ‘ZZ’, the following parameters are available (source order preserved):

| SKU    | Revenue | Demand | Initial Inventory |
|--------|---------|--------|------------------|
| ZZ2AO  | 24.38   | 2      | 10.0             |
| ZZDW7  | 30.12   | 4      | 20.0             |
| ZZM1A  | 19.52   | 82     | 530.0            |
| ZZNC5  | 10.79   | 2      | 10.0             |
| ZZX6K  | 111.81  | 2      | 10.0             |

##### Complete Model

Let $\mathcal{ZZ}$ be the set of all products classified as ‘ZZ’ (here, the five products above). For each $i \in \mathcal{ZZ}$:

$$
\begin{align*}
\max \quad & 24.38\, x_{\text{ZZ2AO}} + 30.12\, x_{\text{ZZDW7}} + 19.52\, x_{\text{ZZM1A}} + 10.79\, x_{\text{ZZNC5}} + 111.81\, x_{\text{ZZX6K}} \\
\text{s.t.} \quad & x_{\text{ZZ2AO}} \leq 2 \\
& x_{\text{ZZ2AO}} \leq 10 \\
& x_{\text{ZZDW7}} \leq 4 \\
& x_{\text{ZZDW7}} \leq 20 \\
& x_{\text{ZZM1A}} \leq 82 \\
& x_{\text{ZZM1A}} \leq 530 \\
& x_{\text{ZZNC5}} \leq 2 \\
& x_{\text{ZZNC5}} \leq 10 \\
& x_{\text{ZZX6K}} \leq 2 \\
& x_{\text{ZZX6K}} \leq 10 \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{ZZ}
\end{align*}
$$

Or, equivalently, for each $i \in \mathcal{ZZ}$:

$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad x_i \in \mathbb{Z}
$$

where the coefficients and bounds are as retrieved above.