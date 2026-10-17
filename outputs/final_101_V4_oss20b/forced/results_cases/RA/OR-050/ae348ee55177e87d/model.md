**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J} \text{Value}_{j}\;x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{j\in J}\text{Weight}_{j}\;x_{ij}\;\le\;\text{Capacity}_{i},
\qquad \forall\,i\in I \\[4pt]
&\sum_{i\in I}x_{i,\text{Smartphone}}\;\ge\;5 \\[4pt]
&x_{ij}\in\mathbb{Z}_{\ge0},\qquad \forall\,i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `ShelfID` |
| \(\text{Capacity}_{i}\) | `file_0_view_0` | `Capacity` |
| \(J\) | `file_1_view_0` | `ProductName` |
| \(\text{Value}_{j}\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_{j}\) | `file_1_view_0` | `Weight` |
| `Smartphone` (first product) | `file_1_view_1` | `ProductName` |

All parameters are taken directly from the corresponding CSV views; no values are hard‑coded in the model. The variable \(x_{ij}\) represents the integer number of units of product \(j\) placed on shelf \(i\). The first‑product constraint uses the product name `"Smartphone"` obtained from the filtered view `file_1_view_1`.