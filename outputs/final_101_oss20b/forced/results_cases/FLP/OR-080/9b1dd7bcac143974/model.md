##### Decision Variables

$x_{it} \geq 0$: weight transported by truck $i \in I$ in period $t \in T$ (continuous).  
$y_i^t \in \{0,1\}$: truck $i$ is active in period $t$.  
$z_i^t \in \{0,1\}$: truck $i$ is started in period $t$.

##### Objective Function

$\displaystyle \min \sum_{i\in I}\sum_{t\in T} C_i\,x_{it}+\sum_{i\in I}\sum_{t\in T} S_i\,z_i^t$

##### Constraints

1. **Demand satisfaction**  
   $\displaystyle \sum_{i\in I}x_{it}\;\geq\;d_t,\quad \forall t\in T$

2. **Spare‑capacity buffer**  
   $\displaystyle \sum_{i\in I}x_{it}\;\leq\;0.9\sum_{i\in I}Q_i\,y_i^t,\quad \forall t\in T$

3. **Truck capacity**  
   $\displaystyle x_{it}\;\leq\;Q_i\,y_i^t,\quad \forall i\in I,\;\forall t\in T$

4. **Weight change limit**  
   $\displaystyle x_{it}-x_{i,t-1}\;\leq\;300,\quad \forall i\in I,\;\forall t\in T,\;t>1$  
   $\displaystyle x_{i,t-1}-x_{it}\;\leq\;300,\quad \forall i\in I,\;\forall t\in T,\;t>1$  
   (with $x_{i0}=0$ for all $i$)

5. **Activation–startup linkage**  
   $\displaystyle y_i^t\;\geq\;z_i^t,\quad \forall i\in I,\;\forall t\in T$  
   $\displaystyle y_i^t\;\leq\;y_i^{t-1}+z_i^t,\quad \forall i\in I,\;\forall t\in T,\;t>1$

6. **Minimum two‑period activity**  
   $\displaystyle y_i^{t+1}\;\geq\;z_i^t,\quad \forall i\in I,\;\forall t\in T,\;t<4$  
   $\displaystyle y_i^{t+2}\;\geq\;z_i^{t+1},\quad \forall i\in I,\;\forall t\in T,\;t<3$

7. **Shutdown separation**  
   $\displaystyle y_i^{t+1}\;\leq\;y_i^t,\quad \forall i\in I,\;\forall t\in T,\;t<4$  
   $\displaystyle y_i^{t+2}\;\leq\;y_i^t,\quad \forall i\in I,\;\forall t\in T,\;t<3$

8. **No start in period 4**  
   $\displaystyle z_i^4=0,\quad \forall i\in I$

9. **Domains**  
   $x_{it}\geq0$ (continuous), $y_i^t,z_i^t\in\{0,1\}$.

**Index sets**  
$I=\{1,2,\dots,10\}$, $T=\{1,2,3,4\}$.

##### Retrieved Information

```
{
  "trucks": [
    {"id":1,"Q":1000,"S":500,"C":2.0},
    {"id":2,"Q":800,"S":300,"C":3.0},
    {"id":3,"Q":1200,"S":400,"C":2.5},
    {"id":4,"Q":600,"S":250,"C":3.0},
    {"id":5,"Q":900,"S":450,"C":2.2},
    {"id":6,"Q":700,"S":280,"C":2.8},
    {"id":7,"Q":1100,"S":420,"C":2.4},
    {"id":8,"Q":500,"S":200,"C":3.2},
    {"id":9,"Q":1000,"S":480,"C":2.1},
    {"id":10,"Q":650,"S":260,"C":2.9}
  ],
  "demand": [1500,2000,1800,1000],
  "Q_vector": [1000,800,1200,600,900,700,1100,500,1000,650],
  "S_vector": [500,300,400,250,450,280,420,200,480,260],
  "C_vector": [2.0,3.0,2.5,3.0,2.2,2.8,2.4,3.2,2.1,2.9]
}
```