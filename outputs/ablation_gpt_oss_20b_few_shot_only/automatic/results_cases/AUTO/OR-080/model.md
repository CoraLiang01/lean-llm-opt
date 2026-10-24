**Mathematical Model**

Sets  
\[
T=\{1,\dots ,10\}\quad\text{(trucks)} ,\qquad  
P=\{1,2,3,4\}\quad\text{(periods)} .
\]

Parameters (from *parameters.csv*)  

\[
\begin{aligned}
Q_t &\;=\;\text{maximum capacity of truck }t,\\
S_t &\;=\;\text{startup cost of truck }t,\\
C_t &\;=\;\text{unit transportation cost of truck }t,\\
d_p &\;=\;\text{customer demand in period }p .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_{t,p}&\in\{0,1\}\quad &&\text{truck }t\text{ active in period }p,\\
s_{t,p}&\in\{0,1\}\quad &&\text{truck }t\text{ started in period }p,\\
w_{t,p}&\ge 0\quad &&\text{weight transported by truck }t\text{ in period }p .
\end{aligned}
\]

Auxiliary constants  

\[
y_{t,0}=0,\qquad w_{t,0}=0\quad\forall t\in T .
\]

---

### Objective
\[
\min \;\sum_{t\in T}\sum_{p\in P}\bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\bigr)
\]

---

### Constraints  

1. **Startup definition**  
   \[
   \begin{aligned}
   &s_{t,1}\;\ge\;y_{t,1},\\
   &s_{t,p}\;\ge\;y_{t,p}-y_{t,p-1}\quad(p=2,3,4),\\
   &s_{t,p}\;\le\;y_{t,p}\quad\forall p,\\
   &s_{t,p}\;\le\;1-y_{t,p-1}\quad(p=2,3,4).
   \end{aligned}
   \]

2. **No start in period 4**  
   \[
   s_{t,4}=0\quad\forall t\in T .
   \]

3. **At least two consecutive periods of activity**  
   \[
   s_{t,p}\;\le\;y_{t,p+1}\quad(p=1,2,3).
   \]

4. **Shutdown rule**  
   \[
   y_{t,p-1}-y_{t,p}\;\le\;1-y_{t,p+1}\quad(p=2,3).
   \]

5. **Capacity limits**  
   \[
   w_{t,p}\;\le\;Q_t\,y_{t,p}\quad\forall t,p .
   \]

6. **Weight‑change limits**  
   \[
   \begin{aligned}
   &w_{t,1}\;\le\;300,\\
   &w_{t,p}-w_{t,p-1}\;\le\;300,\quad
     w_{t,p-1}-w_{t,p}\;\le\;300\quad(p=2,3,4).
   \end{aligned}
   \]

7. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\;d_p\quad\forall p\in P .
   \]

8. **Spare‑capacity buffer (90 %)**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\sum_{t\in T} Q_t\,y_{t,p}\quad\forall p\in P .
   \]

9. **Non‑negativity**  
   \[
   w_{t,p}\;\ge\;0\quad\forall t,p .
   \]

10. **Binary variables**  
    \[
    y_{t,p},\,s_{t,p}\;\in\{0,1\}\quad\forall t,p .
    \]

---

## Data Mapping

| File | Path | Columns | Mapping |
|------|------|---------|----------|
| *parameters.csv* | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv` | `truck_id`, `Q`, `S`, `C`, `d1`, `d2`, `d3`, `d4` | `Q_t = Q[truck_id]`, `S_t = S[truck_id]`, `C_t = C[truck_id]`, `d_p = d{p}[truck_id]` (for each period) |

All parameters are taken directly from the CSV; no additional filtering or aggregation is applied. The model uses the full set of 10 trucks and 4 periods as defined in the data.