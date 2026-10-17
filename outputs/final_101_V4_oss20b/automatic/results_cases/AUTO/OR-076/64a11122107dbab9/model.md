**Mathematical model**

Let  

* \(I=\{W1,\dots ,W10\}\) (warehouse identifiers)  
* \(J=\{C1,\dots ,C20\}\) (customer identifiers)

Parameters  

* \(f_i\) fixed annual opening cost of warehouse \(i\)  
* \(cap_i\) maximum service capacity of warehouse \(i\)  
* \(d_j\) annual demand of customer \(j\)  
* \(c_{ij}\) unit transportation cost from warehouse \(i\) to customer \(j\)

Decision variables  

* \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\)  
* \(y_i\in\{0,1\}\) 1 if warehouse \(i\) is opened, 0 otherwise  

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \qquad\text{(demand satisfaction)}\\[2pt]
&\sum_{j\in J} x_{ij}\le cap_i\,y_i , &&\forall i\in I \qquad\text{(capacity & activation)}\\[2pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J\\
&y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Row key | Column key |
|-----------|--------------|---------|------------|
| \(f_i\) | `file_1_view_0` | `Warehouse ID` | `Fixed_Cost` |
| \(cap_i\) | `file_1_view_0` | `Warehouse ID` | `Capacity` |
| \(d_j\) | `file_2_view_0` | `Customer ID` | `Demand` |
| \(c_{ij}\) | `file_0_view_0` | `Warehouse ID` | `Customer ID` |

*Index sets*  
- \(I\) is the set of all `Warehouse ID` values in `file_1_view_0`.  
- \(J\) is the set of all `Customer ID` values in `file_2_view_0`.  

These mappings provide the exact source of each parameter value; no numerical values are enumerated in the model.