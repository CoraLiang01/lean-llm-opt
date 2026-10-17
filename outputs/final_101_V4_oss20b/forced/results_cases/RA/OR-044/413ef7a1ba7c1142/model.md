**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of sections (SectionID).  
- \(J=\{j\}\) be the set of products (ProductName).  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad\text{(available space in section }i)\\
\text{Value}_j &\quad\text{(revenue per unit of product }j)\\
\text{Weight}_j &\quad\text{(space required per unit of product }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad\forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

---

**Constraints**

1. **Section capacity**

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad\forall i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| Capacity\(_i\) | `file_0_view_0` | `Capacity` |
| Value\(_j\) | `file_1_view_0` | `Value` |
| Weight\(_j\) | `file_1_view_0` | `Weight` |

All indices \(i\) and \(j\) are taken from the corresponding `SectionID` and `ProductName` columns of the supplied tables. No additional constraints or parameters are introduced.