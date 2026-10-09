Let $I = \{A1, A2, \ldots, A80\}$ index the 80 products.

Let $x_i$ = number of 100 kg units of product $i$ to produce in the month (integer, $\geq 0$).

Let $y_i$ = 1 if product $i$'s production line is activated, 0 otherwise (binary).

Parameters (for each $i \in I$):

- $d_i$ = maximum demand (100 kg units)
- $p_i$ = selling price ($/100 kg$)
- $c_i$ = production cost ($/100 kg$)
- $q_i$ = production quota (max per day, 100 kg units)
- $f_i$ = fixed activation cost ($)
- $b_i$ = minimum batch size (100 kg units)
- $T = 22$ (number of production days in the month)

From the data:

- $d_i$ = value from "Maximum Demand (100 kg units)" row, column $i$
- $p_i$ = value from "Selling Price ($/100 kg)" row, column $i$
- $c_i$ = value from "Production Cost ($/100 kg)" row, column $i$
- $q_i$ = value from "Production Quota (max per day)" row, column $i$
- $f_i$ = value from "Activation Cost ($)" row, column $i$
- $b_i$ = value from "Minimum Batch Size (100 kg units)" row, column $i$

The complete model:

$$
\max \sum_{i \in I} \left[ (p_i - c_i) x_i - f_i y_i \right]
$$

Subject to:

1. Demand constraints:
$$
x_i \leq d_i \qquad \forall i \in I
$$

2. Monthly production quota constraints:
$$
x_i \leq T \cdot q_i \qquad \forall i \in I
$$

3. Minimum batch size and activation coupling:
$$
x_i \geq b_i y_i \qquad \forall i \in I
$$

4. Logical upper bound on $x_i$ if not activated:
$$
x_i \leq M_i y_i \qquad \forall i \in I
$$
where $M_i = \min\{d_i, T q_i\}$ (or simply $d_i$ and $T q_i$ are already enforced above, so this constraint is optional/redundant).

5. Variable domains:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$
$$
y_i \in \{0,1\} \qquad \forall i \in I
$$

Where all parameters $d_i, p_i, c_i, q_i, f_i, b_i$ are as given in the retrieved data, for each product $i$ (A1, ..., A80), and $T = 22$.

All coefficients and identifiers are as in the original files and order.