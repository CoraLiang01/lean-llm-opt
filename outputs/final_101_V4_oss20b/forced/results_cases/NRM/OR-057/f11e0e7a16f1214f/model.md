**Mathematical Model**

Let  

- \(P\) – set of platforms (indexed by the values in column *PlatformID*).  
- \(J\) – set of products (indexed by the values in column *ProductName*).  

Parameters  

\[
\begin{aligned}
&\text{Capacity}_p &&\forall p\in P &&\text{(memory capacity of platform }p)\\
&\text{Value}_j &&\forall j\in J &&\text{(value of product }j)\\
&\text{Weight}_j &&\forall j\in J &&\text{(memory weight of product }j)
\end{aligned}
\]

Decision variables  

\[
x_{p j}\in \mathbb{Z}_{\ge 0}\qquad \forall p\in P,\; j\in J
\]

Objective  

\[
\max \;\sum_{p\in P}\sum_{j\in J}\text{Value}_j\,x_{p j}
\]

Constraints  

\[
\sum_{j\in J}\text{Weight}_j\,x_{p j}\;\le\;\text{Capacity}_p
\qquad \forall p\in P
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Capacity}_p\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

The index sets \(P\) and \(J\) are derived from the distinct values of `PlatformID` and `ProductName` in the respective tables. No additional data or constraints are introduced beyond those specified.