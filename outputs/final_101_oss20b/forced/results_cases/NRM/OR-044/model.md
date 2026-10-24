**Mathematical Model**

Let  

- \(I\) be the set of sections (indexed by `SectionID`).  
- \(J\) be the set of products (indexed by `ProductName`).  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad \forall i\in I &&\text{(display space limit)}\\
\text{Value}_j &\quad \forall j\in J &&\text{(revenue per unit)}\\
\text{Weight}_j &\quad \forall j\in J &&\text{(space per unit)}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

Constraints  

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall i\in I
\]

\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

Indices \(I\) and \(J\) are taken from the distinct values of `SectionID` and `ProductName` in the respective tables.