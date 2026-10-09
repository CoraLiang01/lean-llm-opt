**Sets:**  
$I = \{1,2,3,4,5,6,7\}$

**Parameters:**  
\[
\begin{align*}
A &= [119.144,\ 119.144,\ 120.144,\ 119.744,\ 120.844,\ 121.244,\ 121.244] \\
d &= [30,\ 40,\ 50,\ 50,\ 50,\ 30,\ 30] \\
I &= [200,\ 100,\ 150,\ 250,\ 150,\ 150,\ 200]
\end{align*}
\]

**Variables:**  
$x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**  
\[
\max \sum_{i=1}^7 A_i x_i
\]

**Constraints:**  
\[
\begin{align*}
x_i &\leq d_i, \quad \forall i=1,\ldots,7 \\
x_i &\leq I_i, \quad \forall i=1,\ldots,7 \\
x_i &\geq 0,\ x_i \in \mathbb{Z}, \quad \forall i=1,\ldots,7
\end{align*}
\]

**Explicitly:**
\[
\begin{align*}
\max\ & 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 119.744\, x_4 + 120.844\, x_5 + 121.244\, x_6 + 121.244\, x_7 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 50 \\
& 0 \leq x_4 \leq 50 \\
& 0 \leq x_5 \leq 50 \\
& 0 \leq x_6 \leq 30 \\
& 0 \leq x_7 \leq 30 \\
& x_i \in \mathbb{Z},\ \forall i=1,\ldots,7
\end{align*}
\]