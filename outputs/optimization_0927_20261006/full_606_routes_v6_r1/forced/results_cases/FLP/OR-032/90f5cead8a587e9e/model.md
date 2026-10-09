##### Decision Variables

$x_i \geq 0$: Number of units of product $i$ (where $i$ is a ‘Books’ product) to fulfill demand.

##### Parameters

Let $I$ be the set of ‘Books’ products:
$$
I = \{\text{Books\_15.15},\ \text{Books\_30.3},\ \text{Books\_45.45},\ \text{Books\_60.6},\ \text{Books\_75.75}\}
$$

For each $i \in I$:

- Revenue per unit $r_i$:
  - $r_{\text{Books\_15.15}} = 15.15$
  - $r_{\text{Books\_30.3}} = 30.3$
  - $r_{\text{Books\_45.45}} = 45.45$
  - $r_{\text{Books\_60.6}} = 60.6$
  - $r_{\text{Books\_75.75}} = 75.75$

- Initial Inventory $s_i$:
  - $s_{\text{Books\_15.15}} = 9920.0$
  - $s_{\text{Books\_30.3}} = 20160.0$
  - $s_{\text{Books\_45.45}} = 30000.0$
  - $s_{\text{Books\_60.6}} = 38360.0$
  - $s_{\text{Books\_75.75}} = 51450.0$

- Demand $d_i$:
  - $d_{\text{Books\_15.15}} = 1980$
  - $d_{\text{Books\_30.3}} = 3024$
  - $d_{\text{Books\_45.45}} = 4536$
  - $d_{\text{Books\_60.6}} = 5601$
  - $d_{\text{Books\_75.75}} = 7567$

##### Objective Function

$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

1. Inventory and demand bounds for each product:
   $$
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I
   $$

##### Explicitly, for each product:

- $0 \leq x_{\text{Books\_15.15}} \leq 1980$
- $0 \leq x_{\text{Books\_30.3}} \leq 3024$
- $0 \leq x_{\text{Books\_45.45}} \leq 4536$
- $0 \leq x_{\text{Books\_60.6}} \leq 5601$
- $0 \leq x_{\text{Books\_75.75}} \leq 7567$

##### Complete Model

$$
\begin{align*}
\max\quad & 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Books\_15.15}} \leq 1980 \\
& 0 \leq x_{\text{Books\_30.3}} \leq 3024 \\
& 0 \leq x_{\text{Books\_45.45}} \leq 4536 \\
& 0 \leq x_{\text{Books\_60.6}} \leq 5601 \\
& 0 \leq x_{\text{Books\_75.75}} \leq 7567 \\
& x_i \geq 0,\quad \forall i \in I
\end{align*}
$$

Where $x_i$ are continuous variables representing the fulfilled units of each ‘Books’ product.