**Mathematical Model**

Let  

- \(I=\{1,\dots ,10\}\) ( shelves )  
- \(J=\{1,\dots ,20\}\) ( products )  

Parameters  

- \(\text{cap}_i\) – capacity of shelf \(i\)  \(i\in I\)  
- \(\text{val}_j\) – value of product \(j\)  \(j\in J\)  
- \(\text{wt}_j\) – weight of product \(j\)  \(j\in J\)  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\) \(i\in I,\;j\in J\)

---

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J}\text{val}_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J}\text{wt}_j\,x_{ij}\;\le\;\text{cap}_i,
   && \forall\,i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
   && \forall\,i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{val}_j\) | `file_1_view_0` | `Value` |
| \(\text{wt}_j\) | `file_1_view_0` | `Weight` |

| Variable | Domain |
|----------|--------|
| \(x_{ij}\) | integer, \(\ge 0\) |

*Indices \(i\) and \(j\) are taken from the full set of rows returned in the observation tables.*