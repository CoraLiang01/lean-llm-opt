**Mathematical Model**

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks  
* \(P=\{1,2,3,4\}\) – set of time periods  

Parameters (all taken from *parameters.csv*)  

\[
\begin{aligned}
Q_t &\quad\text{(maximum capacity of truck }t)\\
S_t &\quad\text{(startup cost of truck }t)\\
C_t &\quad\text{(unit transport cost of truck }t)\\
d_p &\quad\text{(customer demand in period }p)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
s_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ started in period }p)\\
a_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ active in period }p)\\
r_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ shut down in period }p)\\
w_{t,p} &\ge 0\quad\text{(weight transported by truck }t\text{ in period }p)
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr)
\]

---

### Constraints  

1. **Startup limits**  
   \[
   s_{t,4}=0,\qquad
   s_{t,p}\le 1-a_{t,p-1}\;\;(p=1,\dots ,4)
   \]

2. **Activation dynamics**  
   \[
   a_{t,p}\ge s_{t,p}\;\;(p=1,\dots ,4),\qquad
   a_{t,p}\le a_{t,p-1}+s_{t,p}\;\;(p=2,\dots ,4)
   \]

3. **Minimum two‑period activity**  
   \[
   a_{t,p+1}\ge s_{t,p}\;\;(p=1,2,3)
   \]

4. **Shutdown definition**  
   \[
   r_{t,p}\ge a_{t,p-1}-a_{t,p}\;\;(p=2,\dots ,4),\qquad
   r_{t,p}\le a_{t,p-1}\;\;(p=2,\dots ,4),\qquad
   r_{t,p}\le 1-a_{t,p}\;\;(p=2,\dots ,4)
   \]

5. **Post‑shutdown inactivity**  
   \[
   a_{t,p+1}\le 1-r_{t,p}\;\;(p=2,3)
   \]

6. **Capacity limits**  
   \[
   w_{t,p}\le Q_t\,a_{t,p}\;\;(t\in T,p\in P),\qquad
   w_{t,p}\ge 0
   \]

7. **Weight‑change limits**  
   \[
   w_{t,1}\le 300,\qquad
   |w_{t,p}-w_{t,p-1}|\le 300\;\;(p=2,\dots ,4)
   \]

8. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; d_p\;\;(p\in P)
   \]

9. **Spare‑capacity buffer**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\sum_{t\in T} Q_t\,a_{t,p}\;\;(p\in P)
   \]

---

## Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(Q_t\) | `file_0_view_0` | `Q` |
| \(S_t\) | `file_0_view_0` | `S` |
| \(C_t\) | `file_0_view_0` | `C` |
| \(d_1\) | `file_0_view_0` | `d1` |
| \(d_2\) | `file_0_view_0` | `d2` |
| \(d_3\) | `file_0_view_0` | `d3` |
| \(d_4\) | `file_0_view_0` | `d4` |

All numeric values are taken directly from the corresponding columns of the single
row in `parameters.csv` (the same demand values appear in every row).  
The index sets \(T\) and \(P\) are defined by the 10 truck identifiers and the
four periods, respectively.