Let $x_i$ denote the number of units of Organ product $i$ to fulfill, for each subcategory $i$.

Objective:
$$
\max\ 60.8\, x_{\text{Organic Fruits}} + 918.45\, x_{\text{Organic Staples}} + 77.52\, x_{\text{Organic Vegetables}}
$$

Subject to:
\[
\begin{align*}
& 0 \leq x_{\text{Organic Fruits}} \leq \min\{678906,\ 5034020.0\} \\
& 0 \leq x_{\text{Organic Staples}} \leq \min\{749927,\ 5589290.0\} \\
& 0 \leq x_{\text{Organic Vegetables}} \leq \min\{699808,\ 5202710.0\} \\
& x_{\text{Organic Fruits}},\ x_{\text{Organic Staples}},\ x_{\text{Organic Vegetables}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_i$ = units of Organ product $i$ fulfilled
- Revenue, Demand, and Initial Inventory are as follows (source order):

| Sub Category         | Revenue | Demand  | Initial Inventory |
|---------------------|---------|---------|------------------|
| Organic Fruits      | 60.8    | 678906  | 5034020.0        |
| Organic Staples     | 918.45  | 749927  | 5589290.0        |
| Organic Vegetables  | 77.52   | 699808  | 5202710.0        |