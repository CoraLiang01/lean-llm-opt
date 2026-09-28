Let $I$ be the set of products classified as ‘ELE-S’:

\[
I = \{
\text{ELE-SMA-10000463},
\text{ELE-SMA-10000487},
\text{ELE-SMA-10003333},
\text{ELE-SMA-10009012},
\text{ELE-SMA-10009999},
\text{ELE-SMA-10011234},
\text{ELE-SMA-10027456},
\text{ELE-SMA-10028567}
\}
\]

Let $x_i$ denote the number of units of product $i \in I$ to fulfill (decision variable, $x_i \geq 0$ and $x_i \leq$ integer upper bounds; domain is continuous unless otherwise specified).

Parameters (from source order):

\[
\begin{array}{llll}
\text{Product Reference} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{ELE-SMA-10000463} & 4.0 & 295 & 2000.0 \\
\text{ELE-SMA-10000487} & 14.0 & 1002 & 7000.0 \\
\text{ELE-SMA-10003333} & 14.0 & 958 & 7000.0 \\
\text{ELE-SMA-10009012} & 4.0 & 777 & 6000.0 \\
\text{ELE-SMA-10009999} & 4.0 & 271 & 2000.0 \\
\text{ELE-SMA-10011234} & 4.0 & 244 & 2000.0 \\
\text{ELE-SMA-10027456} & 14.0 & 990 & 7000.0 \\
\text{ELE-SMA-10028567} & 14.0 & 1000 & 7000.0 \\
\end{array}
\]

Model:

Maximize total revenue:
\[
\max \left(
4.0\,x_{\text{ELE-SMA-10000463}}
+ 14.0\,x_{\text{ELE-SMA-10000487}}
+ 14.0\,x_{\text{ELE-SMA-10003333}}
+ 4.0\,x_{\text{ELE-SMA-10009012}}
+ 4.0\,x_{\text{ELE-SMA-10009999}}
+ 4.0\,x_{\text{ELE-SMA-10011234}}
+ 14.0\,x_{\text{ELE-SMA-10027456}}
+ 14.0\,x_{\text{ELE-SMA-10028567}}
\right)
\]

Subject to, for each product $i$:

\[
\begin{align*}
x_{\text{ELE-SMA-10000463}} &\leq 295 \\
x_{\text{ELE-SMA-10000463}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10000463}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10000487}} &\leq 1002 \\
x_{\text{ELE-SMA-10000487}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10000487}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10003333}} &\leq 958 \\
x_{\text{ELE-SMA-10003333}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10003333}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10009012}} &\leq 777 \\
x_{\text{ELE-SMA-10009012}} &\leq 6000.0 \\
x_{\text{ELE-SMA-10009012}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10009999}} &\leq 271 \\
x_{\text{ELE-SMA-10009999}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10009999}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10011234}} &\leq 244 \\
x_{\text{ELE-SMA-10011234}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10011234}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10027456}} &\leq 990 \\
x_{\text{ELE-SMA-10027456}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10027456}} &\geq 0 \\
\\
x_{\text{ELE-SMA-10028567}} &\leq 1000 \\
x_{\text{ELE-SMA-10028567}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10028567}} &\geq 0 \\
\end{align*}
\]

Or, equivalently, for each $i$:
\[
0 \leq x_i \leq \min\{\text{Demand}_i,\,\text{Initial Inventory}_i\}
\]

All variables $x_i$ are continuous and nonnegative (integer if required by context).