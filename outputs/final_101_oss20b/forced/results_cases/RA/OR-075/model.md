**Mathematical Model**

Let  

- \(P\) be the set of all project identifiers extracted from *project.csv* (110 projects).  
- For each \(p\in P\)  
  - \(c_p\) = Capital requirement (k$)  
  - \(v_p\) = Expected NPV (k$)  

Decision variable  

- \(y_p \in \{0,1\}\) (1 if project \(p\) is selected, 0 otherwise)

---

**Objective**

\[
\max \;\sum_{p\in P} v_p\,y_p
\]

---

**Constraints**

1. **Budget**

\[
\sum_{p\in P} c_p\,y_p \;\le\; 1000
\]

2. **Mutually exclusive (Projects 4 and 7)**  

\[
y_{4} + y_{7} \;\le\; 1
\]

3. **Prerequisite (Project 6 requires 1)**  

\[
y_{6} \;\le\; y_{1}
\]

4. **Contingent (Project 10 requires 5)**  

\[
y_{10} \;\le\; y_{5}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_p\) | `file_0_view_0` | `"Capital (k$)"` |
| \(v_p\) | `file_0_view_0` | `"NPV (k$)"` |

| Variable | Domain |
|----------|--------|
| \(y_p\) | binary \(\{0,1\}\) |

*All indices \(p\) are taken from the `"Project ID"` column of `file_0_view_0`.*