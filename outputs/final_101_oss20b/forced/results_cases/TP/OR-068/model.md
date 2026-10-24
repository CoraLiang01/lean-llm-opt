**Mathematical Model**

Let  

- \(M=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
- \(P=\{P1,P2,P3,P4,P5,P6\}\) (Projects)  

Parameters  

- \(c_{mp}\) – assignment cost of manager \(m\) to project \(p\) (from the CSV).  

Decision variables  

- \(x_{mp}\in\{0,1\}\) – 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t.}\;&\sum_{p\in P} x_{mp}=1 &&\forall m\in M \quad\text{(each manager gets one project)}\\
&\sum_{m\in M} x_{mp}=1 &&\forall p\in P \quad\text{(each project gets one manager)}\\
&x_{mp}\in\{0,1\} &&\forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(c_{mp}\) | Table **file_0_view_0** | For each row where `Unnamed: 0` = manager \(m\) and each column `P1`–`P6` = project \(p\), assign the cell value to \(c_{mp}\). |

*The table columns `P1`–`P6` contain the numeric costs; the row identifier `Unnamed: 0` provides the manager key.*