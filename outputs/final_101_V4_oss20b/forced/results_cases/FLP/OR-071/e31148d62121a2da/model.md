**Mathematical Model**

Let  

* \(P\) – set of all products (identified by *Product Name*).  
* \(x_p\) – production quantity of product \(p\) (continuous, \(x_p\ge 0\)).  

Parameters (obtained from *41.csv*)  

* \(L_p\) – labor required per unit of product \(p\).  
* \(M_p\) – material required per unit of product \(p\).  
* \(S_p\) – selling price per unit of product \(p\).  
* \(V_p\) – variable cost per unit of product \(p\).  

Fixed data  

* \(C_{\text{lab}}\;=\;1650\) (labor capacity per week).  
* \(C_{\text{mat}}\;=\;1850\) (material capacity per week).  
* \(F_{\text{fixed}}\;=\;4500\) (fixed operating cost per week).  

---

\[
\begin{aligned}
\max_{x}\quad & \sum_{p\in P}\bigl(S_p-V_p\bigr)\,x_p \;-\; F_{\text{fixed}} \\[4pt]
\text{s.t.}\quad
& \sum_{p\in P} L_p\,x_p \;\le\; C_{\text{lab}} \qquad &&\text{(labor)}\\[4pt]
& \sum_{p\in P} M_p\,x_p \;\le\; C_{\text{mat}} \qquad &&\text{(material)}\\[4pt]
& x_p \;\ge\; 0 \qquad &&\forall\,p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column in Table |
|-----------|--------------|-----------------|
| \(L_p\) | `file_0_view_0` | `Labor per unit` |
| \(M_p\) | `file_0_view_0` | `Material per unit` |
| \(S_p\) | `file_0_view_0` | `Selling Price` |
| \(V_p\) | `file_0_view_0` | `Variable Cost` |
| \(C_{\text{lab}}\) | *fixed* | constant 1650 |
| \(C_{\text{mat}}\) | *fixed* | constant 1850 |
| \(F_{\text{fixed}}\) | *fixed* | constant 4500 |

*Index set* \(P\) is the set of all distinct values in the `Product Name` column of `file_0_view_0`. No other data are required.