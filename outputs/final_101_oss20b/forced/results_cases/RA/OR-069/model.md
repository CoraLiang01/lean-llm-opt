**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{m,p}=1 && \forall m\in M \quad\text{(each manager assigned once)}\\
& \sum_{m\in M} x_{m,p}=1 && \forall p\in P \quad\text{(each project assigned once)}\\[4pt]
& x_{m,p}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

**Data Mapping**

| Table ID | Column | Parameter / Variable |
|----------|--------|----------------------|
| `file_0_view_0` | `Manager` | Manager identifier \(m\) |
| `file_0_view_0` | `Project 1 Cost` | \(c_{m,1}\) |
| `file_0_view_0` | `Project 2 Cost` | \(c_{m,2}\) |
| `file_0_view_0` | `Project 3 Cost` | \(c_{m,3}\) |
| `file_0_view_0` | `Project 4 Cost` | \(c_{m,4}\) |
| `file_0_view_0` | `Project 5 Cost` | \(c_{m,5}\) |
| `file_0_view_0` | `Project 6 Cost` | \(c_{m,6}\) |
| `file_0_view_0` | `Project 7 Cost` | \(c_{m,7}\) |
| `file_0_view_0` | `Project 8 Cost` | \(c_{m,8}\) |
| `file_0_view_0` | `Project 9 Cost` | \(c_{m,9}\) |
| `file_0_view_0` | `Project 10 Cost` | \(c_{m,10}\) |
| `file_0_view_0` | `Project 11 Cost` | \(c_{m,11}\) |

- **Parameter** \(c_{m,p}\) is taken from the column *Project \(p\) Cost* for the row where *Manager* = \(m\).
- **Variable** \(x_{m,p}\) is binary, indicating whether manager \(m\) is assigned to project \(p\).