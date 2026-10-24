**Mathematical Model**

```math
\textbf{Sets}
\begin{aligned}
I &:= \{\,i \mid \text{product } i \text{ appears in the filtered table}\,\}
\end{aligned}

\textbf{Parameters}
\begin{aligned}
r_i &:= \text{revenue per unit of product } i \\
d_i &:= \text{demand quantity of product } i \\
s_i &:= \text{initial inventory of product } i
\end{aligned}

\textbf{Decision Variables}
\begin{aligned}
x_i &\in \mathbb{Z}_{+} \quad \text{(units of product } i \text{ to sell)}
\end{aligned}

\textbf{Objective}
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

\textbf{Constraints}
\[
\begin{aligned}
0 &\le x_i \le d_i &&\forall i\in I \\
0 &\le x_i \le s_i &&\forall i\in I
\end{aligned}
\]
```

---

**Data Mapping Rules**

| Table ID | Column | Parameter / Index |
|----------|--------|-------------------|
| `file_0_view_0` | `Product Name` | Index `i` (unique product identifier) |
| `file_0_view_0` | `Revenue` | `r_i` |
| `file_0_view_0` | `Demand` | `d_i` |
| `file_0_view_0` | `Initial Inventory` | `s_i` |

*The index set `I` is constructed from all rows that satisfy the filter condition (`Product Name` contains `"TABLET"`). Each row supplies the corresponding parameter values for that index.*