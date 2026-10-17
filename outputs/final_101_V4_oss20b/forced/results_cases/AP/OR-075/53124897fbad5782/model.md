**Mathematical Model**

Let  

- \(P=\{1,2,\dots ,110\}\) be the set of projects.  
- \(x_p\in\{0,1\}\) be a decision variable that equals 1 if project \(p\) is selected.  

Parameters (obtained from *project.csv*)  

- \(c_p\) – Capital investment of project \(p\) (k$).  
- \(v_p\) – Expected Net Present Value of project \(p\) (k$).  

**Objective**

\[
\max \;\sum_{p\in P} v_p\,x_p
\]

**Constraints**

1. **Budget**

\[
\sum_{p\in P} c_p\,x_p \;\le\; 1000
\]

2. **Mutually exclusive (Projects 4 and 7)**  

\[
x_4 + x_7 \;\le\; 1
\]

3. **Prerequisite (Project 6 requires Project 1)**  

\[
x_6 \;\le\; x_1
\]

4. **Contingent (Project 10 requires Project 5)**  

\[
x_{10} \;\le\; x_5
\]

5. **Binary variables**

\[
x_p \in \{0,1\}\quad \forall p\in P
\]

---

**Data Mapping**

| Parameter | Source Table | Column in Table | Mapping |
|-----------|--------------|-----------------|---------|
| \(c_p\)   | file_0_view_0 | “Capital (k$)”  | \(c_p =\) value in row where “Project ID” = \(p\) |
| \(v_p\)   | file_0_view_0 | “NPV (k$)”      | \(v_p =\) value in row where “Project ID” = \(p\) |

*Table ID:* `file_0_view_0`  
*Columns used:* `"Project ID"`, `"Capital (k$)"`, `"NPV (k$)"`  

All other data are implicit in the index set \(P\).