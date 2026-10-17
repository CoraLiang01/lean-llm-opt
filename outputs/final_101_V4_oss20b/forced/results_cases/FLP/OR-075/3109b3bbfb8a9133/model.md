**Mathematical Model**

Let  

- \(I=\{1,\dots ,110\}\) : set of projects.  
- \(x_i\in\{0,1\}\) : 1 if project \(i\) is selected, 0 otherwise.  

Parameters (obtained from *project.csv*, table id **file_0_view_0**):

- \(c_i\) : capital requirement of project \(i\) (k$).  
- \(p_i\) : expected NPV of project \(i\) (k$).  

Constant:

- \(B=1000\) : total capital budget (k$).

\[
\begin{aligned}
\max\;&\sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } &\sum_{i\in I} c_i\,x_i \;\le\; B \qquad\text{(Budget)}\\[4pt]
&x_4 + x_7 \;\le\; 1 \qquad\text{(Mutually exclusive)}\\[4pt]
&x_6 \;\le\; x_1 \qquad\text{(Prerequisite)}\\[4pt]
&x_{10} \;\le\; x_5 \qquad\text{(Contingent)}\\[4pt]
&x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name | Mapping |
|-----------|--------------|-------------|---------|
| \(I\) | *project.csv* | *Project ID* | index set \(I=\{1,\dots,110\}\) |
| \(c_i\) | *project.csv* | *Capital (k$)* | \(c_i =\) value in column *Capital (k$)* for project \(i\) |
| \(p_i\) | *project.csv* | *NPV (k$)* | \(p_i =\) value in column *NPV (k$)* for project \(i\) |

All other constants and constraints are defined explicitly in the model above. No additional data or enumeration of rows is required.