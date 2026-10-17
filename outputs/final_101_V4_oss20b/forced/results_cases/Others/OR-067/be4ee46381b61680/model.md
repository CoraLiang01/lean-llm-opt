**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{s.t.}\;&\sum_{p\in P} x_{m,p}=1 &&\forall m\in M \\[2pt]
&\sum_{m\in M} x_{m,p}=1 &&\forall p\in P \\[2pt]
&x_{m,p}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

*Indices*  
\(M=\{\text{MA, MB, MC}\}\) – managers (rows of the cost table)  
\(P=\{\text{P1, P2, P3}\}\) – projects (columns of the cost table)

*Parameters*  
\(c_{m,p}\) – cost for manager \(m\) to handle project \(p\) (taken from the table)

*Decision variables*  
\(x_{m,p}=1\) if manager \(m\) is assigned to project \(p\); otherwise \(0\).

---

**Data Mapping**

| Table ID | Column | Mapping to Model |
|----------|--------|------------------|
| `file_0_view_0` | `Unnamed: 0` | Manager identifier → index set \(M\) |
| `file_0_view_0` | `P1`, `P2`, `P3` | Project identifiers → index set \(P\) |
| `file_0_view_0` | `P1`, `P2`, `P3` (rows) | Cell value at row *m*, column *p* → parameter \(c_{m,p}\) |

*Notes*  
- The index sets \(M\) and \(P\) are exactly the unique values appearing in the `Unnamed: 0` column and the column names `P1`–`P3`, respectively.  
- No additional rows or columns are introduced; the model uses only the data present in the supplied CSV.  
- All decision variables are binary, as required for a one‑to‑one assignment.