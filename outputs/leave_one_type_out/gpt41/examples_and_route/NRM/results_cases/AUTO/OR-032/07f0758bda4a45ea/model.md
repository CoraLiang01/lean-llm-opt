Let $x_i$ be the number of units of product $i$ (where $i$ indexes the five ‘Books’ products below) to be fulfilled.

Objective:
\[
\max \; 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}}
\]

Subject to, for each product $i$:
\[
\begin{align*}
0 &\leq x_{\text{Books\_15.15}} \leq \min\{1980,\; 9920.0\} = 1980 \\
0 &\leq x_{\text{Books\_30.3}} \leq \min\{3024,\; 20160.0\} = 3024 \\
0 &\leq x_{\text{Books\_45.45}} \leq \min\{4536,\; 30000.0\} = 4536 \\
0 &\leq x_{\text{Books\_60.6}} \leq \min\{5601,\; 38360.0\} = 5601 \\
0 &\leq x_{\text{Books\_75.75}} \leq \min\{7567,\; 51450.0\} = 7567 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]

Where:
- $x_i$ = units of product $i$ fulfilled (decision variable, nonnegative integer)
- Revenue, Demand, and Initial Inventory for each product are as follows:

| Product_Name      | Revenue | Demand | Initial Inventory |
|-------------------|---------|--------|------------------|
| Books_15.15       | 15.15   | 1980   | 9920.0           |
| Books_30.3        | 30.3    | 3024   | 20160.0          |
| Books_45.45       | 45.45   | 4536   | 30000.0          |
| Books_60.6        | 60.6    | 5601   | 38360.0          |
| Books_75.75       | 75.75   | 7567   | 51450.0          |