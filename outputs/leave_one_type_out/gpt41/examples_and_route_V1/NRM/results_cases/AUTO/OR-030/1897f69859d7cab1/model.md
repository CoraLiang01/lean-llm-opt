Let $i$ index the car models classified under ‘FDK57’ in the order they appear in the data. Let $x_i$ be the quantity of car model $i$ to fulfill.

Objective:
\[
\max \; 119.144\, x_1 + 121.244\, x_2 + 120.144\, x_3 + 119.144\, x_4 + 120.544\, x_5 + 119.744\, x_6
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{30,\,200\} \\
& 0 \leq x_2 \leq \min\{30,\,200\} \\
& 0 \leq x_3 \leq \min\{50,\,150\} \\
& 0 \leq x_4 \leq \min\{40,\,100\} \\
& 0 \leq x_5 \leq \min\{10,\,150\} \\
& 0 \leq x_6 \leq \min\{50,\,250\} \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,6
\end{align*}
\]

Where:
- $x_i$ = quantity of FDK57 car model $i$ to fulfill (nonnegative integer)
- The upper bound for each $x_i$ is the minimum of its Demand and Initial Inventory, as sales cannot exceed either.

Data (in source order):

| $i$ | Product Name | Revenue   | Demand | Initial Inventory |
|-----|--------------|-----------|--------|------------------|
| 1   | FDK57        | 119.144   | 30     | 200              |
| 2   | FDK57        | 121.244   | 30     | 200              |
| 3   | FDK57        | 120.144   | 50     | 150              |
| 4   | FDK57        | 119.144   | 40     | 100              |
| 5   | FDK57        | 120.544   | 10     | 150              |
| 6   | FDK57        | 119.744   | 50     | 250              |