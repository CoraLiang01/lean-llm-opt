**Symbolic Model**

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks  
* \(P=\{1,2,3,4\}\) – set of periods  

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
a_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ active in period }p)\\
s_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ started in period }p)\\
r_{t,p} &\in\{0,1\}\quad\text{(truck }t\text{ shut down in period }p)\\
w_{t,p} &\ge 0\quad\text{(weight transported by truck }t\text{ in period }p)
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr)
\]

---

### Constraints  

1. **Startup–activation linkage**  
   \[
   a_{t,p}\;\ge\; s_{t,p}\qquad \forall t,p
   \]

2. **No start in period 4**  
   \[
   s_{t,4}=0 \qquad \forall t
   \]

3. **Shutdown definition**  
   \[
   r_{t,p}\;\ge\; a_{t,p-1}-a_{t,p}\qquad \forall t,\;p=2,\dots ,4
   \]
   \[
   r_{t,p}\;\le\; a_{t,p-1}\qquad \forall t,p
   \]
   \[
   r_{t,p}\;\le\; 1-a_{t,p}\qquad \forall t,p
   \]

4. **Post‑shutdown inactivity**  
   \[
   a_{t,p+1}\;\le\; 1-r_{t,p}\qquad \forall t,\;p=1,2,3
   \]
   \[
   a_{t,p+2}\;\le\; 1-r_{t,p}\qquad \forall t,\;p=1,2
   \]

5. **No restart before two periods after shutdown**  
   \[
   s_{t,p}\;\le\; 1-r_{t,p-1}\qquad \forall t,\;p=2,\dots ,4
   \]

6. **Initial state**  
   \[
   a_{t,0}=0,\; s_{t,0}=0,\; r_{t,0}=0,\; w_{t,0}=0 \qquad \forall t
   \]

7. **Capacity limits**  
   \[
   w_{t,p}\;\le\; Q_t\,a_{t,p}\qquad \forall t,p
   \]

8. **Weight‑change limits**  
   \[
   w_{t,p}\;\le\; w_{t,p-1}+300 \qquad \forall t,\;p=2,\dots ,4
   \]
   \[
   w_{t,p-1}\;\le\; w_{t,p}+300 \qquad \forall t,\;p=2,\dots ,4
   \]
   (For \(p=1\): \(w_{t,1}\le 300\))

9. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; d_p \qquad \forall p
   \]

10. **Spare‑capacity buffer (90 % of active capacity)**  
    \[
    \sum_{t\in T} w_{t,p}\;\le\; 0.9\;\sum_{t\in T} Q_t\,a_{t,p}\qquad \forall p
    \]

---

**Data Mapping**

| Table ID | Column in CSV | Parameter / Variable |
|----------|---------------|----------------------|
| file_0_view_0 | truck_id | \(t \in T\) |
| file_0_view_0 | Q | \(Q_t\) |
| file_0_view_0 | S | \(S_t\) |
| file_0_view_0 | C | \(C_t\) |
| file_0_view_0 | d1 | \(d_1\) |
| file_0_view_0 | d2 | \(d_2\) |
| file_0_view_0 | d3 | \(d_3\) |
| file_0_view_0 | d4 | \(d_4\) |

All numerical values for the parameters are taken directly from the rows of *parameters.csv* as listed above. No additional data are introduced.