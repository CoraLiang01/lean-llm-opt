**Mathematical Model**

Let  

- \(P\) be the set of all project identifiers (the 110 values in column **Project ID**).  
- \(y_p \in \{0,1\}\) be a binary decision variable that equals 1 if project \(p\) is selected, 0 otherwise.  

Parameters (all taken directly from the CSV view **file_0_view_0**):

- \(C_p\) – capital requirement of project \(p\) (column **Capital (k$)**).  
- \(N_p\) – expected NPV of project \(p\) (column **NPV (k$)**).  

The model is:

\[
\begin{aligned}
\max\;& \sum_{p\in P} N_p\,y_p \\[4pt]
\text{s.t.}\;& \sum_{p\in P} C_p\,y_p \;\le\; 1000 \\[4pt]
& y_4 + y_7 \;\le\; 1 \\[4pt]
& y_6 \;\le\; y_1 \\[4pt]
& y_{10} \;\le\; y_5 \\[4pt]
& y_p \in \{0,1\}\quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(P\) | `file_0_view_0` | `Project ID` |
| \(C_p\) | `file_0_view_0` | `Capital (k$)` |
| \(N_p\) | `file_0_view_0` | `NPV (k$)` |

All parameters are read directly from the corresponding columns of the CSV view; no values are enumerated in the model.