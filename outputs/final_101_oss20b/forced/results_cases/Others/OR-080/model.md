**Mathematical Model**

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks (truck\_id).  
* \(P=\{1,2,3,4\}\) – set of periods.  

Parameters (all taken from *file_0_view_0*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(Q_t\) | maximum capacity of truck \(t\) (kg) | `Q` |
| \(S_t\) | startup cost of truck \(t\) (yuan) | `S` |
| \(C_t\) | unit transportation cost of truck \(t\) (yuan/kg) | `C` |
| \(d_p\) | customer demand in period \(p\) (kg) | `d1`–`d4` |

Decision variables  

* \(x_{t,p}\in\{0,1\}\) – 1 if truck \(t\) is active in period \(p\).  
* \(s_{t,p}\in\{0,1\}\) – 1 if truck \(t\) starts in period \(p\).  
* \(w_{t,p}\ge 0\) – weight transported by truck \(t\) in period \(p\) (kg).

---

### Objective  

\[
\min \;\sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr)
\]

---

### Constraints  

1. **Startup definition**  
   \[
   \begin{aligned}
   &s_{t,1}=x_{t,1}\qquad &&\forall t\in T\\
   &s_{t,p}\le x_{t,p}\qquad &&\forall t\in T,\;p=2,\dots ,4\\
   &s_{t,p}\le 1-x_{t,p-1}\qquad &&\forall t\in T,\;p=2,\dots ,4\\
   &s_{t,p}\ge x_{t,p}-x_{t,p-1}\qquad &&\forall t\in T,\;p=2,\dots ,4
   \end{aligned}
   \]

2. **No start in period 4**  
   \[
   s_{t,4}=0\qquad\forall t\in T
   \]

3. **At least two consecutive active periods**  
   \[
   s_{t,p}\le x_{t,p+1}\qquad\forall t\in T,\;p=1,2,3
   \]

4. **Shutdown and restart rules**  
   \[
   \begin{aligned}
   &x_{t,p}-x_{t,p+1}\le x_{t,p+2}\qquad &&\forall t\in T,\;p=1,2\\
   &x_{t,p+2}\le x_{t,p+1}+x_{t,p}\qquad &&\forall t\in T,\;p=1,2
   \end{aligned}
   \]

5. **Capacity limits**  
   \[
   w_{t,p}\le Q_t\,x_{t,p}\qquad\forall t\in T,\;p\in P
   \]

6. **Weight‑change limits**  
   \[
   \begin{aligned}
   &w_{t,p}-w_{t,p-1}\le 300\qquad &&\forall t\in T,\;p=1,\dots ,4\\
   &w_{t,p-1}-w_{t,p}\le 300\qquad &&\forall t\in T,\;p=1,\dots ,4
   \end{aligned}
   \]
   (with \(w_{t,0}=0\) for all \(t\))

7. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; d_p\qquad\forall p\in P
   \]

8. **10 % spare‑capacity buffer**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\;\sum_{t\in T} Q_t\,x_{t,p}\qquad\forall p\in P
   \]

9. **Variable domains**  
   \[
   \begin{aligned}
   &x_{t,p}\in\{0,1\}\qquad &&\forall t\in T,\;p\in P\\
   &s_{t,p}\in\{0,1\}\qquad &&\forall t\in T,\;p\in P\\
   &w_{t,p}\ge 0\qquad &&\forall t\in T,\;p\in P
   \end{aligned}
   \]

---

### Data Mapping

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(Q_t\) | `file_0_view_0` | `Q` | \(Q_t =\) value in row where `truck_id = t` |
| \(S_t\) | `file_0_view_0` | `S` | \(S_t =\) value in row where `truck_id = t` |
| \(C_t\) | `file_0_view_0` | `C` | \(C_t =\) value in row where `truck_id = t` |
| \(d_p\) | `file_0_view_0` | `d1`–`d4` | \(d_1 =\) `d1`, \(d_2 =\) `d2`, \(d_3 =\) `d3`, \(d_4 =\) `d4` |

All other data (demands, capacities, costs) are taken directly from the corresponding columns of the single source file *parameters.csv* (view `file_0_view_0`). No additional data or assumptions are introduced.