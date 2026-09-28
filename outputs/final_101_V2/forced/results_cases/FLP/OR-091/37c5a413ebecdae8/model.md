##### Decision Variables

Let $x_i \in \{0,1\}$ indicate whether Operations Research course $i$ is selected.

##### Parameters

Let $I$ be the set of Operations Research courses:
$$
I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}
$$

Credits for each course:
\[
\begin{align*}
\text{C22}: &\quad 5 \\
\text{C23}: &\quad 5 \\
\text{C24}: &\quad 4 \\
\text{C25}: &\quad 4 \\
\text{C26}: &\quad 4 \\
\text{C27}: &\quad 4 \\
\text{C28}: &\quad 4 \\
\end{align*}
\]

Interest points for each course:
\[
\begin{align*}
\text{C22}: &\quad 95 \\
\text{C23}: &\quad 92 \\
\text{C24}: &\quad 86 \\
\text{C25}: &\quad 82 \\
\text{C26}: &\quad 85 \\
\text{C27}: &\quad 80 \\
\text{C28}: &\quad 88 \\
\end{align*}
\]

##### Objective Function

\[
\max \sum_{i \in I} p_i x_i
\]
where $p_i$ is the interest points for course $i$.

Explicitly:
\[
\max\ 95x_{\text{C22}} + 92x_{\text{C23}} + 86x_{\text{C24}} + 82x_{\text{C25}} + 85x_{\text{C26}} + 80x_{\text{C27}} + 88x_{\text{C28}}
\]

##### Constraints

1. Credit limit:
\[
\sum_{i \in I} c_i x_i \leq 20
\]
where $c_i$ is the credits for course $i$.

Explicitly:
\[
5x_{\text{C22}} + 5x_{\text{C23}} + 4x_{\text{C24}} + 4x_{\text{C25}} + 4x_{\text{C26}} + 4x_{\text{C27}} + 4x_{\text{C28}} \leq 20
\]

2. Binary selection:
\[
x_i \in \{0,1\},\quad \forall i \in I
\]

##### Complete Model

\[
\begin{align*}
\max\ & 95x_{\text{C22}} + 92x_{\text{C23}} + 86x_{\text{C24}} + 82x_{\text{C25}} + 85x_{\text{C26}} + 80x_{\text{C27}} + 88x_{\text{C28}} \\
\text{s.t.}\quad & 5x_{\text{C22}} + 5x_{\text{C23}} + 4x_{\text{C24}} + 4x_{\text{C25}} + 4x_{\text{C26}} + 4x_{\text{C27}} + 4x_{\text{C28}} \leq 20 \\
& x_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

Where $I = \{\text{C22},\ \text{C23},\ \text{C24},\ \text{C25},\ \text{C26},\ \text{C27},\ \text{C28}\}$, with credits and interest points as listed above.