Let $x_i$ be the number of units of product $i$ to fulfill for customer purchases.

Let $I$ be the set of products, indexed by their "Product Name" as given in the data.

Parameters for each product $i \in I$:
- $r_i$: Revenue per unit (from "Revenue")
- $d_i$: Demand (from "Demand")
- $s_i$: Initial Inventory (from "Initial Inventory")

The complete mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & \sum_{i \in I} r_i x_i \\[2ex]
\text{subject to} \quad
& 0 \leq x_i \leq \min\{d_i,\, s_i\} \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$

Where:

- $x_i$: Number of units of product $i$ fulfilled for customer purchases (decision variable, nonnegative integer)
- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory for product $i$

All parameters are taken directly from the retrieved data, preserving the original product names and values.

Explicitly, for each product $i$:

- $r_i$ = value in "Revenue" column
- $d_i$ = value in "Demand" column
- $s_i$ = value in "Initial Inventory" column

The model maximizes total revenue by choosing, for each product, how many units to fulfill, subject to not exceeding either the available inventory or the demand for that product. All variables are nonnegative integers.