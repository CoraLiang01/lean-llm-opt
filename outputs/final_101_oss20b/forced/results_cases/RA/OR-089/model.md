**Mathematical Model**

Let  

* \(S=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(C=\{C1,C2,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_j\) – fixed opening cost of centre \(j\)  \(j\in S\)  
* \(c_{ij}\) – cost of serving customer \(i\) from centre \(j\)  \(i\in C,\;j\in S\)

Decision variables  

* \(y_j\in\{0,1\}\) \(j\in S\) (1 if centre \(j\) is opened)  
* \(x_{ij}\in\{0,1\}\) \(i\in C,\;j\in S\) (1 if customer \(i\) is assigned to centre \(j\))

\[
\begin{aligned}
\min\;&\sum_{j\in S} f_j\,y_j \;+\; \sum_{i\in C}\sum_{j\in S} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{j\in S} x_{ij}=1, &&\forall i\in C \qquad\text{(each customer served once)}\\[2pt]
&x_{ij}\le y_j, &&\forall i\in C,\;j\in S \qquad\text{(assignment only to open centre)}\\[2pt]
&\sum_{i\in C} x_{ij}\le 4, &&\forall j\in S \qquad\text{(capacity of centre)}\\[2pt]
&x_{ij}\in\{0,1\},\;y_j\in\{0,1\}, &&\forall i\in C,\;j\in S
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Key |
|-----------|----------|-----------|-----|
| \(f_j\) | `file_1_view_0` | `Fixed Opening Cost` | `Service Center` |
| \(c_{ij}\) | `file_0_view_0` | `SC1` … `SC10` | `Customer` |

*The table IDs and column names are taken verbatim from the CSVQA observation.*