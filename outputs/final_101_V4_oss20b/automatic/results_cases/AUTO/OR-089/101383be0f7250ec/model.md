**Mathematical Model**

Let  

* \(I=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(J=\{C1,C2,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_i\) – fixed opening cost of centre \(i\)  
* \(c_{ij}\) – cost of serving customer \(j\) from centre \(i\)

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if centre \(i\) is opened, 0 otherwise  
* \(x_{ij} \in \{0,1\}\) – 1 if customer \(j\) is assigned to centre \(i\)

---

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=1, &&\forall j\in J \qquad\text{(each customer served once)}\\[2pt]
& \sum_{j\in J} x_{ij}\;\le\;4\,y_i, &&\forall i\in I \qquad\text{(capacity of opened centre)}\\[2pt]
& x_{ij}\in\{0,1\}, &&\forall i\in I,\;j\in J\\
& y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(f_i\) | `file_1_view_0` | `Service Center` | `Fixed Opening Cost` | \(f_i =\) value in row where `Service Center` = \(i\) |
| \(c_{ij}\) | `file_0_view_0` | `Customer` (row) | `SC1`–`SC10` (columns) | \(c_{ij} =\) value in row where `Customer` = \(j\) and column `SCi` |

*All indices \(i\) and \(j\) are taken from the full sets of rows and columns in the respective tables; no additional filtering is applied.*