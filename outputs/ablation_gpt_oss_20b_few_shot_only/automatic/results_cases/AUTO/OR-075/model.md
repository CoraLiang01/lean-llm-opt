**Model**

Let  

* \(P=\{1,2,\dots ,110\}\) – set of all projects.  
* \(x_p\in\{0,1\}\) – decision variable, 1 if project \(p\) is selected.  

Parameters (data from *project.csv*)  

* \(c_p\) – capital requirement of project \(p\) (k$).  
* \(n_p\) – expected NPV of project \(p\) (k$).  

Objective  

\[
\max \;\sum_{p\in P} n_p\,x_p
\]

Constraints  

1. **Budget**  
\[
\sum_{p\in P} c_p\,x_p \;\le\; 1000
\]

2. **Mutual exclusivity (4 & 7)**  
\[
x_4 + x_7 \;\le\; 1
\]

3. **Prerequisite (6 requires 1)**  
\[
x_6 \;\le\; x_1
\]

4. **Contingent (10 requires 5)**  
\[
x_{10} \;\le\; x_5
\]

5. **Binary variables**  
\[
x_p \in \{0,1\}\quad \forall p\in P
\]

---

**Data Mapping**

| Parameter | Source file | Column name | Mapping expression |
|-----------|-------------|-------------|---------------------|
| \(c_p\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv` | `Capital (k$)` | \(c_p =\) value in row where `Project ID = p` |
| \(n_p\) | same file | `NPV (k$)` | \(n_p =\) value in row where `Project ID = p` |

All constraints refer to the same index set \(P\) and use the parameters as mapped above. No additional data or assumptions are introduced.