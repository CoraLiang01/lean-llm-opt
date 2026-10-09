Let $I$ be the set of products classified as ‘27in’:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Let $x_i$ be the number of units of product $i \in I$ to fulfill.

Parameters:
\[
\begin{align*}
\text{Revenue}_{\text{27in 4K Gaming Monitor}} &= 389.99 \\
\text{Revenue}_{\text{27in FHD Monitor}} &= 149.99 \\
\text{Demand}_{\text{27in 4K Gaming Monitor}} &= 12474 \\
\text{Demand}_{\text{27in FHD Monitor}} &= 15057 \\
\text{InitialInventory}_{\text{27in 4K Gaming Monitor}} &= 62440 \\
\text{InitialInventory}_{\text{27in FHD Monitor}} &= 75500 \\
\end{align*}
\]

Decision variables:
\[
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\]

Objective:
\[
\max\ 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}}
\]

Subject to:
\[
\begin{align*}
& x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
& x_{\text{27in 4K Gaming Monitor}} \leq 62440 \\
& x_{\text{27in FHD Monitor}} \leq 15057 \\
& x_{\text{27in FHD Monitor}} \leq 75500 \\
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \geq 0,\ \text{integer}
\end{align*}
\]

Or, equivalently (since demand is less than inventory for both products):
\[
\begin{align*}
& 0 \leq x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
& 0 \leq x_{\text{27in FHD Monitor}} \leq 15057 \\
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z}
\end{align*}
\]