Let $I$ be the set of products classified as ‘27in’:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Let $x_i$ be the number of units of product $i \in I$ to be fulfilled.

Parameters (from the data):

\[
\begin{array}{l|ccc}
\text{Product Name} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{27in 4K Gaming Monitor} & 261.2933 & 12474 & 62440 \\
\text{27in FHD Monitor} & 52.4965 & 15057 & 75500 \\
\end{array}
\]

Objective:
\[
\max\ 261.2933\, x_{\text{27in 4K Gaming Monitor}} + 52.4965\, x_{\text{27in FHD Monitor}}
\]

Subject to:
\[
\begin{align*}
0 \leq x_{\text{27in 4K Gaming Monitor}} &\leq \min\{12474,\ 62440\} = 12474 \\
0 \leq x_{\text{27in FHD Monitor}} &\leq \min\{15057,\ 75500\} = 15057 \\
x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} &\in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Or, more generally for all $i \in I$:
\[
\begin{align*}
\max\ & \sum_{i \in I} \text{Revenue}_i\, x_i \\
\text{s.t.}\quad & 0 \leq x_i \leq \min\{\text{Demand}_i,\ \text{Initial Inventory}_i\},\quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]