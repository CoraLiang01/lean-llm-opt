##### Decision Variables

$x_{i,t}\geq0$: weight transported by truck $i\in I$ in period $t\in T$ (continuous).  
$y_{i,t}\in\{0,1\}$: truck $i$ is active in period $t$ (binary).  
$z_{i,t}\in\{0,1\}$: truck $i$ is started in period $t$ (binary).  

##### Objective Function

$\displaystyle \min\sum_{i\in I}\sum_{t\in T}\bigl(C_i\,x_{i,t}+S_i\,z_{i,t}\bigr)$  

##### Constraints  

1. **Activation dynamics**  
   $y_{i,1}=z_{i,1}\quad\forall i\in I$  
   $y_{i,t}=y_{i,t-1}+z_{i,t}\quad\forall i\in I,\;t=2,3,4$  

2. **Minimum two‑period active rule**  
   $y_{i,2}\geq z_{i,1}\quad\forall i\in I$  

3. **No start in period 4**  
   $z_{i,4}=0\quad\forall i\in I$  

4. **Shutdown persistence**  
   $y_{i,t+1}\leq y_{i,t}\quad\forall i\in I,\;t=1,2,3$  

5. **Restart after shutdown**  
   $z_{i,t}\leq 1-y_{i,t-1}\quad\forall i\in I,\;t=2,3,4$  

6. **Capacity limits**  
   $0\leq x_{i,t}\leq Q_i\,y_{i,t}\quad\forall i\in I,\;t\in T$  

7. **Weight change limits**  
   $x_{i,1}\leq 300\quad\forall i\in I$  
   $|x_{i,t}-x_{i,t-1}|\leq 300\quad\forall i\in I,\;t=2,3,4$  

8. **Demand satisfaction**  
   $\displaystyle \sum_{i\in I}x_{i,t}\geq D_t\quad\forall t\in T$  

9. **Spare‑capacity buffer**  
   $\displaystyle \sum_{i\in I}x_{i,t}\leq 0.9\sum_{i\in I}Q_i\,y_{i,t}\quad\forall t\in T$  

10. **Binary and non‑negativity**  
    $y_{i,t}\in\{0,1\},\;z_{i,t}\in\{0,1\},\;x_{i,t}\geq0\quad\forall i\in I,\;t\in T$  

$T=\{1,2,3,4\}$, $I=\{1,2,3,4,5,6,7,8,9,10\}$, $D_1=1500,\;D_2=2000,\;D_3=1800,\;D_4=1000$.

##### Retrieved Information

```json
{
  "trucks": [
    {"truck_id":"1","Q":"1000","S":"500","C":"2.0"},
    {"truck_id":"2","Q":"800","S":"300","C":"3.0"},
    {"truck_id":"3","Q":"1200","S":"400","C":"2.5"},
    {"truck_id":"4","Q":"600","S":"250","C":"3.0"},
    {"truck_id":"5","Q":"900","S":"450","C":"2.2"},
    {"truck_id":"6","Q":"700","S":"280","C":"2.8"},
    {"truck_id":"7","Q":"1100","S":"420","C":"2.4"},
    {"truck_id":"8","Q":"500","S":"200","C":"3.2"},
    {"truck_id":"9","Q":"1000","S":"480","C":"2.1"},
    {"truck_id":"10","Q":"650","S":"260","C":"2.9"}
  ],
  "period_demands": {
    "1":"1500",
    "2":"2000",
    "3":"1800",
    "4":"1000"
  }
}
```