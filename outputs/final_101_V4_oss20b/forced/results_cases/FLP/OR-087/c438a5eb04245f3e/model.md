**Mathematical model**

Let  

* \(P=\{1,\dots ,80\}\) – product indices  
* \(D=\{1,\dots ,22\}\) – day indices  

Decision variables  

\[
\begin{aligned}
x_{p,d}&\in \mathbb{Z}_{\ge 0}\qquad &&\text{units of product }p\text{ produced on day }d\\
y_{p}&\in\{0,1\}\qquad &&\text{product }p\text{ is activated (at least one day)}\\
z_{p,d}&\in\{0,1\}\qquad &&\text{product }p\text{ is produced on day }d
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
D_{\max p}&\ge 0 &&\text{maximum demand for product }p\text{ (over the month)}\\
p_{p}&\ge 0 &&\text{selling price }(\$ /100\text{ kg})\\
c_{p}&\ge 0 &&\text{production cost }(\$ /100\text{ kg})\\
q_{p}&\ge 0 &&\text{daily production quota for product }p\\
b_{p}&\ge 0 &&\text{minimum batch size for product }p\\
f_{p}&\ge 0 &&\text{fixed activation cost for product }p
\end{aligned}
\]

---

### Objective  

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(p_{p}-c_{p}\bigr)\,x_{p,d}\;-\;\sum_{p\in P}f_{p}\,y_{p}
\]

---

### Constraints  

1. **Demand (upper bound)**  
   \[
   \sum_{d\in D}x_{p,d}\;\le\;D_{\max p}\qquad\forall p\in P
   \]

2. **Daily quota**  
   \[
   x_{p,d}\;\le\;q_{p}\,z_{p,d}\qquad\forall p\in P,\;d\in D
   \]

3. **Minimum batch size**  
   \[
   x_{p,d}\;\ge\;b_{p}\,z_{p,d}\qquad\forall p\in P,\;d\in D
   \]

4. **Activation linkage**  
   \[
   z_{p,d}\;\le\;y_{p}\qquad\forall p\in P,\;d\in D
   \]

5. **Variable domains**  
   \[
   x_{p,d}\in\mathbb{Z}_{\ge0},\; y_{p}\in\{0,1\},\; z_{p,d}\in\{0,1\}
   \]

---

## Data Mapping

| Parameter | Source table | Source row | Source column | Mapping rule |
|-----------|--------------|------------|---------------|--------------|
| \(D_{\max p}\) | `36-1.csv` | 0 | `A{p}` | `D_{\max p} = value in row 0, column A{p}` |
| \(p_{p}\) | `36-1.csv` | 1 | `A{p}` | `p_{p} = value in row 1, column A{p}` |
| \(c_{p}\) | `36-1.csv` | 2 | `A{p}` | `c_{p} = value in row 2, column A{p}` |
| \(q_{p}\) | `36-1.csv` | 3 | `A{p}` | `q_{p} = value in row 3, column A{p}` |
| \(f_{p}\) | `36-2.csv` | 0 | `A{p}` | `f_{p} = value in row 0, column A{p}` |
| \(b_{p}\) | `36-3.csv` | 0 | `A{p}` | `b_{p} = value in row 0, column A{p}` |

All indices \(p\) run from 1 to 80, and \(d\) from 1 to 22. The model uses only the data specified above; no additional parameters are introduced.