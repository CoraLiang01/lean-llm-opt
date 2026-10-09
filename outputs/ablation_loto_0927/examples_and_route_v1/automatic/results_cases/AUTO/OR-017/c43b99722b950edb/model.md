Let $x_i$ be the number of units of product $i$ (where $i$ is a SKU classified as ‘ZZ’) to fulfill.

Objective:
$$
\max \; 24.38\, x_{\text{ZZ2AO}} + 30.12\, x_{\text{ZZDW7}} + 19.52\, x_{\text{ZZM1A}} + 10.79\, x_{\text{ZZNC5}} + 111.81\, x_{\text{ZZX6K}}
$$

Subject to:
\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq \min\{2,\,10.0\} = 2 \\
0 \leq x_{\text{ZZDW7}} &\leq \min\{4,\,20.0\} = 4 \\
0 \leq x_{\text{ZZM1A}} &\leq \min\{82,\,530.0\} = 82 \\
0 \leq x_{\text{ZZNC5}} &\leq \min\{2,\,10.0\} = 2 \\
0 \leq x_{\text{ZZX6K}} &\leq \min\{2,\,10.0\} = 2 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{ZZ2AO}, \text{ZZDW7}, \text{ZZM1A}, \text{ZZNC5}, \text{ZZX6K}\}
\end{align*}
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $0 \leq x_i \leq \min\{\text{Demand}_i,\,\text{Initial Inventory}_i\}$)
- Revenue, Demand, and Initial Inventory are as given in the table below:

| SKU     | Revenue | Demand | Initial Inventory |
|---------|---------|--------|------------------|
| ZZ2AO   | 24.38   | 2      | 10.0             |
| ZZDW7   | 30.12   | 4      | 20.0             |
| ZZM1A   | 19.52   | 82     | 530.0            |
| ZZNC5   | 10.79   | 2      | 10.0             |
| ZZX6K   | 111.81  | 2      | 10.0             |