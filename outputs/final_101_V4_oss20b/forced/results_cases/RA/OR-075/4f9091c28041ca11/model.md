**Mathematical Model**

Let  

- \(P\) be the set of all projects (project identifiers from the file).  
- \(y_p \in \{0,1\}\) be a binary decision variable that equals 1 if project \(p\) is selected, 0 otherwise.  

Parameters (obtained from the CSV view `file_0_view_0`):

- \(C_p\) – capital requirement of project \(p\) (k$).  
- \(N_p\) – expected Net Present Value of project \(p\) (k$).  

---

\[
\begin{aligned}
\max \quad & \sum_{p\in P} N_p\,y_p \\[4pt]
\text{s.t.}\quad
& \sum_{p\in P} C_p\,y_p \;\le\; 1000 \qquad &&\text{(Budget)}\\[4pt]
& y_{4} + y_{7} \;\le\; 1 \qquad &&\text{(Mutually exclusive)}\\[4pt]
& y_{6} \;\le\; y_{1} \qquad &&\text{(Prerequisite)}\\[4pt]
& y_{10} \;\le\; y_{5} \qquad &&\text{(Contingent)}\\[4pt]
& y_p \in \{0,1\} \qquad &&\forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(C_p\) | `file_0_view_0` | `Capital (k$)` |
| \(N_p\) | `file_0_view_0` | `NPV (k$)` |

The index set \(P\) is derived from the `Project ID` column of the same table. No other data are required.