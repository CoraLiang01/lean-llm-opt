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

Parameters (for each $i \in I$):

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

Decision variables:

\[
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i \in I
\]

Objective:

\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Constraints (for each $i \in I$):

\[
\begin{align*}
x_i &\leq \text{Initial Inventory}_i \\
x_i &\leq \text{Demand}_i \\
x_i &\geq 0
\end{align*}
\]

Explicitly, the model is:

\[
\max\ 
4.0\,x_{\text{ELE-SMA-10000463}}
+ 14.0\,x_{\text{ELE-SMA-10000487}}
+ 14.0\,x_{\text{ELE-SMA-10003333}}
+ 4.0\,x_{\text{ELE-SMA-10009012}}
+ 4.0\,x_{\text{ELE-SMA-10009999}}
+ 4.0\,x_{\text{ELE-SMA-10011234}}
+ 14.0\,x_{\text{ELE-SMA-10027456}}
+ 14.0\,x_{\text{ELE-SMA-10028567}}
\]

Subject to:

\[
\begin{align*}
0 \leq\ &x_{\text{ELE-SMA-10000463}} \leq \min\{2000.0,\ 295\} \\
0 \leq\ &x_{\text{ELE-SMA-10000487}} \leq \min\{7000.0,\ 1002\} \\
0 \leq\ &x_{\text{ELE-SMA-10003333}} \leq \min\{7000.0,\ 958\} \\
0 \leq\ &x_{\text{ELE-SMA-10009012}} \leq \min\{6000.0,\ 777\} \\
0 \leq\ &x_{\text{ELE-SMA-10009999}} \leq \min\{2000.0,\ 271\} \\
0 \leq\ &x_{\text{ELE-SMA-10011234}} \leq \min\{2000.0,\ 244\} \\
0 \leq\ &x_{\text{ELE-SMA-10027456}} \leq \min\{7000.0,\ 990\} \\
0 \leq\ &x_{\text{ELE-SMA-10028567}} \leq \min\{7000.0,\ 1000\} \\
\end{align*}
\]

where each $x_i$ is a nonnegative continuous variable representing the fulfilled quantity of product $i$.