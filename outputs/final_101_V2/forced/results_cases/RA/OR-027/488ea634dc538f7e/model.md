Let $i$ index the Organ products, with the following identifiers and parameters:

- Organic Fruits: Revenue = 60.8, Demand = 678906, Initial Inventory = 5034020.0
- Organic Staples: Revenue = 918.45, Demand = 749927, Initial Inventory = 5589290.0
- Organic Vegetables: Revenue = 77.52, Demand = 699808, Initial Inventory = 5202710.0

Decision variables:
$x_i$ = number of units of Organ product $i$ to fulfill, for each $i$.

Objective:
$$
\max\ 60.8\, x_{\text{Organic Fruits}} + 918.45\, x_{\text{Organic Staples}} + 77.52\, x_{\text{Organic Vegetables}}
$$

Subject to:
\[
\begin{align*}
& x_{\text{Organic Fruits}} \leq 678906 \\
& x_{\text{Organic Fruits}} \leq 5034020.0 \\
& x_{\text{Organic Staples}} \leq 749927 \\
& x_{\text{Organic Staples}} \leq 5589290.0 \\
& x_{\text{Organic Vegetables}} \leq 699808 \\
& x_{\text{Organic Vegetables}} \leq 5202710.0 \\
& x_{\text{Organic Fruits}},\ x_{\text{Organic Staples}},\ x_{\text{Organic Vegetables}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_i$ is the number of units of Organ product $i$ fulfilled (nonnegative integer, for each $i$).
- Each $x_i$ cannot exceed both its demand and its initial inventory.