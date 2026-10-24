**Mathematical Model**

Let  

- \(M=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(P=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{m,p}\) – assignment cost of manager \(m\) to project \(p\).

Decision variables  

- \(x_{m,p}\in\{0,1\}\) (1 if manager \(m\) is assigned to project \(p\), 0 otherwise).

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\
\text{s.t.}\;&\sum_{p\in P} x_{m,p}=1, &&\forall m\in M \quad\text{(each manager gets one project)}\\
&\sum_{m\in M} x_{m,p}=1, &&\forall p\in P \quad\text{(each project gets one manager)}\\
&x_{m,p}\in\{0,1\}, &&\forall m\in M,\;p\in P .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column(s) | Mapping |
|--------|--------|----------|-----------|---------|
| \(M\) | Manager list | `file_0_view_0` | `Unnamed: 0` | Set of distinct values in column `Unnamed: 0`. |
| \(P\) | Project list | `file_0_view_0` | `P1`–`P6` | Set of column names `P1`, `P2`, `P3`, `P4`, `P5`, `P6`. |
| \(c_{m,p}\) | Assignment cost | `file_0_view_0` | `P1`–`P6` | For each row with manager `m` (value in `Unnamed: 0`) and column `p`, the numeric entry is \(c_{m,p}\). |

No other data are required. The model uses only the information supplied in the CSV file.