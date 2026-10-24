**Mathematical Model**

Let  

\[
I=\{1,2,\dots ,110\}
\]

be the set of all projects.  
For each \(i\in I\) define

* \(c_i\) – capital requirement of project \(i\) (k$)  
* \(p_i\) – expected NPV of project \(i\) (k$)  
* \(x_i\in\{0,1\}\) – binary decision variable (1 = select project \(i\))  

The objective is to maximize total expected NPV:

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

subject to

1. **Budget constraint**

\[
\sum_{i\in I} c_i\,x_i \;\le\; 1000
\]

2. **Mutually exclusive (projects 4 and 7)**

\[
x_4 + x_7 \;\le\; 1
\]

3. **Prerequisite (project 6 requires 1)**

\[
x_6 \;\le\; x_1
\]

4. **Contingent (project 10 requires 5)**

\[
x_{10} \;\le\; x_5
\]

5. **Binary decision variables**

\[
x_i \in \{0,1\}\quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_i\) | Capital (k$) | file_0_view_0 | Capital (k$) |
| \(p_i\) | NPV (k$) | file_0_view_0 | NPV (k$) |
| \(x_i\) | Decision variable | – | – |

*All 110 rows of `project.csv` are used; no filtering is applied.*  

The constraints above reference the parameters exactly as they appear in the source table. No additional data or assumptions are introduced.