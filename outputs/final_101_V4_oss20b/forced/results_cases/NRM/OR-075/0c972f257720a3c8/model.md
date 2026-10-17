**Mathematical Model**

Let  

- \(I=\{1,2,\dots ,110\}\) be the set of projects.  
- \(c_i\) = Capital requirement of project \(i\) (k$).  
- \(v_i\) = Expected Net Present Value of project \(i\) (k$).  
- \(x_i\in\{0,1\}\) = 1 if project \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} c_i\,x_i \;\le\; 1000 \\[4pt]
& x_4 + x_7 \;\le\; 1 \\[4pt]
& x_6 \;\le\; x_1 \\[4pt]
& x_{10} \;\le\; x_5 \\[4pt]
& x_i \in \{0,1\}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capital (k$)` |
| \(v_i\)   | `file_0_view_0` | `NPV (k$)` |
| \(x_i\)   | Decision variable | — |

All parameter values are taken directly from the rows of `project.csv` (view `file_0_view_0`). No additional data or assumptions are introduced beyond what is specified in the query.