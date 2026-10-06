Let $I$ be the set of all TABLET products:

\[
I = \{
\text{TABLET\_10084.74},
\text{TABLET\_12211.86},
\text{TABLET\_14669.5},
\text{TABLET\_14745.76},
\text{TABLET\_14754.24},
\text{TABLET\_16448.3},
\text{TABLET\_16448.31},
\text{TABLET\_20143.22},
\text{TABLET\_2042.38},
\text{TABLET\_24915.25},
\text{TABLET\_24915.26},
\text{TABLET\_26448.3},
\text{TABLET\_27042.38},
\text{TABLET\_30000.0},
\text{TABLET\_33397.46},
\text{TABLET\_33398.3},
\text{TABLET\_48567.8},
\text{TABLET\_48644.07},
\text{TABLET\_50262.72},
\text{TABLET\_53736.44},
\text{TABLET\_6957.62},
\text{TABLET\_6957.63},
\text{TABLET\_7550.84},
\text{TABLET\_7550.85},
\text{TABLET\_9584.74},
\text{TABLET\_9661.02},
\text{TABLET\_9669.5}
\}
\]

Let $x_i$ be the number of units of TABLET product $i$ to fulfill, for each $i \in I$.

Parameters for each $i \in I$ (from the data):

\[
\begin{array}{llll}
\text{Product Name} & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
\hline
\text{TABLET\_10084.74} & 10084.74 & 2 & 10 \\
\text{TABLET\_12211.86} & 12211.86 & 43 & 300 \\
\text{TABLET\_14669.5} & 14669.5 & 6 & 30 \\
\text{TABLET\_14745.76} & 14745.76 & 6 & 30 \\
\text{TABLET\_14754.24} & 14754.24 & 20 & 100 \\
\text{TABLET\_16448.3} & 16448.3 & 22 & 110 \\
\text{TABLET\_16448.31} & 16448.31 & 3 & 20 \\
\text{TABLET\_20143.22} & 20143.22 & 16 & 80 \\
\text{TABLET\_2042.38} & 2042.38 & 2 & 10 \\
\text{TABLET\_24915.25} & 24915.25 & 2 & 10 \\
\text{TABLET\_24915.26} & 24915.26 & 32 & 160 \\
\text{TABLET\_26448.3} & 26448.3 & 14 & 70 \\
\text{TABLET\_27042.38} & 27042.38 & 2 & 10 \\
\text{TABLET\_30000.0} & 30000.0 & 2 & 10 \\
\text{TABLET\_33397.46} & 33397.46 & 6 & 30 \\
\text{TABLET\_33398.3} & 33398.3 & 2 & 10 \\
\text{TABLET\_48567.8} & 48567.8 & 6 & 30 \\
\text{TABLET\_48644.07} & 48644.07 & 2 & 10 \\
\text{TABLET\_50262.72} & 50262.72 & 6 & 30 \\
\text{TABLET\_53736.44} & 53736.44 & 6 & 30 \\
\text{TABLET\_6957.62} & 6957.62 & 6 & 40 \\
\text{TABLET\_6957.63} & 6957.63 & 12 & 80 \\
\text{TABLET\_7550.84} & 7550.84 & 60 & 300 \\
\text{TABLET\_7550.85} & 7550.85 & 8 & 40 \\
\text{TABLET\_9584.74} & 9584.74 & 8 & 40 \\
\text{TABLET\_9661.02} & 9661.02 & 38 & 190 \\
\text{TABLET\_9669.5} & 9669.5 & 4 & 20 \\
\end{array}
\]

Objective:
\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to, for all $i \in I$:
\[
0 \leq x_i \leq \min(\text{Demand}_i, \text{InitialInventory}_i)
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each $i$:

\[
\begin{align*}
0 \leq x_{\text{TABLET\_10084.74}} &\leq 2 \\
0 \leq x_{\text{TABLET\_12211.86}} &\leq 43 \\
0 \leq x_{\text{TABLET\_14669.5}} &\leq 6 \\
0 \leq x_{\text{TABLET\_14745.76}} &\leq 6 \\
0 \leq x_{\text{TABLET\_14754.24}} &\leq 20 \\
0 \leq x_{\text{TABLET\_16448.3}} &\leq 22 \\
0 \leq x_{\text{TABLET\_16448.31}} &\leq 3 \\
0 \leq x_{\text{TABLET\_20143.22}} &\leq 16 \\
0 \leq x_{\text{TABLET\_2042.38}} &\leq 2 \\
0 \leq x_{\text{TABLET\_24915.25}} &\leq 2 \\
0 \leq x_{\text{TABLET\_24915.26}} &\leq 32 \\
0 \leq x_{\text{TABLET\_26448.3}} &\leq 14 \\
0 \leq x_{\text{TABLET\_27042.38}} &\leq 2 \\
0 \leq x_{\text{TABLET\_30000.0}} &\leq 2 \\
0 \leq x_{\text{TABLET\_33397.46}} &\leq 6 \\
0 \leq x_{\text{TABLET\_33398.3}} &\leq 2 \\
0 \leq x_{\text{TABLET\_48567.8}} &\leq 6 \\
0 \leq x_{\text{TABLET\_48644.07}} &\leq 2 \\
0 \leq x_{\text{TABLET\_50262.72}} &\leq 6 \\
0 \leq x_{\text{TABLET\_53736.44}} &\leq 6 \\
0 \leq x_{\text{TABLET\_6957.62}} &\leq 6 \\
0 \leq x_{\text{TABLET\_6957.63}} &\leq 12 \\
0 \leq x_{\text{TABLET\_7550.84}} &\leq 60 \\
0 \leq x_{\text{TABLET\_7550.85}} &\leq 8 \\
0 \leq x_{\text{TABLET\_9584.74}} &\leq 8 \\
0 \leq x_{\text{TABLET\_9661.02}} &\leq 38 \\
0 \leq x_{\text{TABLET\_9669.5}} &\leq 4 \\
\end{align*}
\]

and all $x_i$ are integer.

Summary:

Maximize
\[
\sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to
\[
0 \leq x_i \leq \min(\text{Demand}_i, \text{InitialInventory}_i), \quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]