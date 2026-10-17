**Mathematical Model**

\[
\begin{aligned}
\text{Sets: } & T=\{1,\dots ,40\}, \qquad M=\{1,2,3\} \\[4pt]
\text{Parameters: } & 
\begin{cases}
\text{BI}_i &\text{basic instructions of task }i\\
f_k &\text{frequency of CPU }k\\
p_{i,k}= \dfrac{\text{BI}_i}{f_k} &\text{processing time of task }i\text{ on CPU }k\\
M_{\text{big}}=\displaystyle\frac{\sum_{i\in T}\text{BI}_i}{\min_{k\in M}f_k} &\text{big‑M constant}
\end{cases} \\[6pt]
\text{Decision variables: } & 
\begin{cases}
x_{i,k}\in\{0,1\} &\text{1 if task }i\text{ is assigned to CPU }k\\
s_i\ge 0 &\text{start time of task }i\\
C_i\ge 0 &\text{completion time of task }i\\
C_{\max}\ge 0 &\text{makespan}\\
y_{i,j,k}\in\{0,1\} &\text{1 if task }i\text{ precedes task }j\text{ on CPU }k\;(i<j)
\end{cases} \\[6pt]
\text{Objective: } & \min \; C_{\max} \\[6pt]
\text{Constraints: } & 
\begin{aligned}
&\sum_{k\in M} x_{i,k}=1 &&\forall i\in T &&\text{(assignment)}\\
&C_i = s_i + \sum_{k\in M} x_{i,k}\,p_{i,k} &&\forall i\in T &&\text{(completion time)}\\
&s_i \ge 0 &&\forall i\in T &&\text{(non‑negativity)}\\
&C_{\max}\ge C_i &&\forall i\in T &&\text{(makespan)}\\
&\text{For all }i<j,\;k\in M:\\
&\quad s_i + p_{i,k}\;\le\; s_j + M_{\text{big}}\bigl(1-y_{i,j,k}\bigr)
      + M_{\text{big}}\bigl(2-x_{i,k}-x_{j,k}\bigr)\\
&\quad s_j + p_{j,k}\;\le\; s_i + M_{\text{big}}\,y_{i,j,k}
      + M_{\text{big}}\bigl(2-x_{i,k}-x_{j,k}\bigr)\\
&\quad y_{i,j,k}\in\{0,1\}
\end{aligned}
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| `BI_i` | Table `file_0_view_0` (column name equals task index `i` as string) | `BI_i = value of column i in the single row of file_0_view_0` |
| `f_k` | Query constants | `f_1 = 1.33`, `f_2 = 2`, `f_3 = 2.66` |
| `p_{i,k}` | Derived | `p_{i,k} = BI_i / f_k` |
| `M_big` | Derived | `M_big = (∑_{i∈T} BI_i) / min(f_k)` |

All other sets (`T`, `M`) and binary variables (`x_{i,k}`, `y_{i,j,k}`) are defined by the model itself. No numerical values are enumerated; the model references the data through the mapping rules above.