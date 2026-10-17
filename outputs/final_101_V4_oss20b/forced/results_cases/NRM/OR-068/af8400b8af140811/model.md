**Mathematical Model**

Let  

- \(M=\{m_1,\dots ,m_6\}\) be the set of managers (rows of the CSV).  
- \(P=\{p_1,\dots ,p_6\}\) be the set of projects (columns \(P1\!-\!P6\)).  

Parameters  

- \(c_{mp}\) – assignment cost of manager \(m\) to project \(p\).  

Decision variables  

- \(x_{mp}\in\{0,1\}\) – equals 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

\[
\begin{aligned}
\min\;& \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t.}\;& \sum_{p\in P} x_{mp}=1 &&\forall m\in M \quad\text{(each manager gets one project)}\\[2pt]
& \sum_{m\in M} x_{mp}=1 &&\forall p\in P \quad\text{(each project gets one manager)}\\[2pt]
& x_{mp}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Mapping Rule |
|--------|--------|--------------|
| \(M\) | `file_0_view_0` | Set of distinct values in column **Unnamed: 0** |
| \(P\) | `file_0_view_0` | Set of column names **P1**–**P6** |
| \(c_{mp}\) | `file_0_view_0` | Value in row where **Unnamed: 0** = \(m\) and column **Pj** = \(p\) |

No numerical values are listed; the model is fully symbolic and ready for instantiation with the supplied CSV data.