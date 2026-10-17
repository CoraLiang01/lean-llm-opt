**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{m,p}=1 &&\forall m\in M \\[2pt]
& \sum_{m\in M} x_{m,p}=1 &&\forall p\in P \\[2pt]
& x_{m,p}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(M\) | `file_0_view_0` | `Manager` |
| \(P\) | – | `{1,2,3,4,5,6,7}` (derived from column names) |
| \(c_{m,p}\) | `file_0_view_0` | `Project {p} Cost` (e.g., `Project 1 Cost`, `Project 2 Cost`, …, `Project 7 Cost`) |
| \(x_{m,p}\) | – | – |

*All indices \(m\) and \(p\) are taken from the corresponding sets \(M\) and \(P\) defined above.*