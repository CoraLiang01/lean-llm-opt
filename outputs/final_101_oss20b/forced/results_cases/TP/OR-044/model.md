**Mathematical model**

Let  

* \(I=\{1,\dots ,8\}\) sections (SectionID)  
* \(J=\{1,\dots ,10\}\) products (ProductName)  

Parameters  

* \(\text{capacity}[i]\) display space available in section \(i\)  
* \(\text{value}[j]\) price (revenue per unit) of product \(j\)  
* \(\text{weight}[j]\) shelf‑space required per unit of product \(j\)

Decision variables  

* \(x_{ij}\in \mathbb{Z}_{\ge 0}\) number of units of product \(j\) placed in section \(i\)

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J}\text{value}[j]\;x_{ij} \\[4pt]
\text{s.t. }&\sum_{j\in J}\text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i],
\qquad \forall i\in I \\[4pt]
&x_{ij}\;\in\;\mathbb{Z}_{\ge 0},\qquad \forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Column | Index column | Mapping rule |
|-----------|--------------|--------|--------------|--------------|
| `capacity[i]` | `file_0_view_0` | `Capacity` | `SectionID` | `capacity[SectionID] = Capacity` |
| `value[j]` | `file_1_view_0` | `Value` | `ProductName` | `value[ProductName] = Value` |
| `weight[j]` | `file_1_view_0` | `Weight` | `ProductName` | `weight[ProductName] = Weight` |

All indices are taken directly from the values in the corresponding columns.  
No additional constraints or bounds are imposed beyond those listed in the model.