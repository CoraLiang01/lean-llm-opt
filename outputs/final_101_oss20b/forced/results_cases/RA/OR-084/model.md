**Mathematical Model**

Let  

* \(I=\{1,\dots ,40\}\) – set of tasks  
* \(J=\{1,2,3\}\) – set of CPUs  

Parameters  

* \(p_i\) – basic instructions (in billions) of task \(i\)  
* \(s_j\) – clock speed of CPU \(j\)  
  \[
  s_1=1.33,\qquad s_2=2,\qquad s_3=2.66
  \]

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if task \(i\) is assigned to CPU \(j\), 0 otherwise  
* \(C_{\max}\ge 0\) – makespan (completion time of the last task)

---

\[
\begin{aligned}
\min\;& C_{\max} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each task assigned once)}\\[4pt]
& \sum_{i\in I}\frac{p_i}{s_j}\,x_{ij}\;\le\; C_{\max} &&\forall j\in J \quad\text{(CPU load)}\\[4pt]
& x_{ij}\in\{0,1\} &&\forall i\in I,\; j\in J\\
& C_{\max}\ge 0
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Column |
|-----------|--------|--------|
| \(p_i\)   | `file_0_view_0` | task identifier (e.g., `"1"`, `"2"`, … `"40"`) |
| \(s_j\)   | – (constant) | – |
| \(C_{\max}\) | – | – |
| \(x_{ij}\) | – | – |

*The values of \(p_i\) are taken directly from the single row of `18.csv`; each column name corresponds to a task ID.*