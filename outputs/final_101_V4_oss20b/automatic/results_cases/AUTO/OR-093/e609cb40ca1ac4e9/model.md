**Mathematical Model**

Let  

- \(M=\{M_1,\dots ,M_{12}\}\) (“Machine” column)  
- \(T=\{A,B,\dots ,L\}\) (task columns)  

Parameters  

- \(c_{i,j}\) machining cost for assigning machine \(i\in M\) to task \(j\in T\).

Decision variables  

- \(x_{i,j}\in\{0,1\}\) equals 1 if machine \(i\) is assigned to task \(j\), 0 otherwise.

\[
\begin{aligned}
\min\;&\sum_{i\in M}\sum_{j\in T} c_{i,j}\,x_{i,j} \\[4pt]
\text{s.t.}\;&\sum_{j\in T} x_{i,j}=1 &&\forall i\in M \\[2pt]
&\sum_{i\in M} x_{i,j}=1 &&\forall j\in T \\[2pt]
&x_{i,j}\in\{0,1\} &&\forall i\in M,\; j\in T
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Mapping to Model |
|----------|--------|------------------|
| `file_0_view_0` | `Machine` | index \(i\in M\) |
| `file_0_view_0` | `A`–`L` | index \(j\in T\) |
| `file_0_view_0` | `A`–`L` values | parameter \(c_{i,j}\) |

All 12 rows and 12 task columns are used; no additional filtering or aggregation is applied.