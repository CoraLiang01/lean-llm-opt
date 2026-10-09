**Decision Variables:**  
For each product $k \in \{\text{P1}, \ldots, \text{P111}\}$:  
$\quad x_k \geq 0$ (continuous), the quantity of product $k$ to produce.

---

**Parameters:**  
- Let $p_k$ = Unit_Profit of product $k$ (from unit_product_profits.csv)
- Let $a_{ik}$ = processing time required by product $k$ on device $i$ (from device_time.csv)
- Let $c_i$ = Monthly_Capacity of device $i$ (from monthly_device_capacity.csv)

---

**Objective:**  
Maximize total profit:
$$
\max \sum_{k=\text{P1}}^{\text{P111}} p_k \, x_k
$$

---

**Constraints:**  
For each device $i \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:
$$
\sum_{k=\text{P1}}^{\text{P111}} a_{ik} \, x_k \leq c_i
$$

Where:
- $a_{ik}$ is the value in row with Device $i$ and column $k$ in device_time.csv
- $c_i$ is the Monthly_Capacity for Device $i$ in monthly_device_capacity.csv
- $p_k$ is the Unit_Profit for Product $k$ in unit_product_profits.csv

---

**Variable Domains:**  
$$
x_k \geq 0 \quad \forall k \in \{\text{P1}, \ldots, \text{P111}\}
$$

---

**Explicit Formulation with Data:**

Let the set of devices be $D = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$  
Let the set of products be $K = \{\text{P1}, \ldots, \text{P111}\}$

**Objective:**
$$
\max \left(
\begin{aligned}
&28.55\,x_{\text{P1}} + 12.78\,x_{\text{P2}} + 45.21\,x_{\text{P3}} + 18.92\,x_{\text{P4}} + 33.47\,x_{\text{P5}} + 8.64\,x_{\text{P6}} + 25.88\,x_{\text{P7}} + 40.15\,x_{\text{P8}} \\
&+ 14.39\,x_{\text{P9}} + 37.62\,x_{\text{P10}} + 10.25\,x_{\text{P11}} + 29.81\,x_{\text{P12}} + 48.77\,x_{\text{P13}} + 16.53\,x_{\text{P14}} + 42.09\,x_{\text{P15}} + 9.11\,x_{\text{P16}} \\
&+ 22.46\,x_{\text{P17}} + 36.73\,x_{\text{P18}} + 11.99\,x_{\text{P19}} + 31.28\,x_{\text{P20}} + 7.45\,x_{\text{P21}} + 27.6\,x_{\text{P22}} + 43.91\,x_{\text{P23}} + 15.82\,x_{\text{P24}} \\
&+ 39.36\,x_{\text{P25}} + 6.23\,x_{\text{P26}} + 20.71\,x_{\text{P27}} + 34.98\,x_{\text{P28}} + 13.24\,x_{\text{P29}} + 32.53\,x_{\text{P30}} + 5.10\,x_{\text{P31}} + 24.18\,x_{\text{P32}} \\
&+ 47.85\,x_{\text{P33}} + 19.30\,x_{\text{P34}} + 41.44\,x_{\text{P35}} + 11.02\,x_{\text{P36}} + 26.57\,x_{\text{P37}} + 38.20\,x_{\text{P38}} + 17.76\,x_{\text{P39}} + 46.12\,x_{\text{P40}} \\
&+ 9.89\,x_{\text{P41}} + 21.34\,x_{\text{P42}} + 33.86\,x_{\text{P43}} + 14.88\,x_{\text{P44}} + 30.41\,x_{\text{P45}} + 6.50\,x_{\text{P46}} + 28.93\,x_{\text{P47}} + 49.20\,x_{\text{P48}} \\
&+ 18.15\,x_{\text{P49}} + 44.78\,x_{\text{P50}} + 10.57\,x_{\text{P51}} + 23.69\,x_{\text{P52}} + 35.51\,x_{\text{P53}} + 16.03\,x_{\text{P54}} + 38.83\,x_{\text{P55}} + 5.88\,x_{\text{P56}} \\
&+ 29.45\,x_{\text{P57}} + 42.33\,x_{\text{P58}} + 12.41\,x_{\text{P59}} + 37.06\,x_{\text{P60}} + 7.99\,x_{\text{P61}} + 21.90\,x_{\text{P62}} + 46.99\,x_{\text{P63}} + 19.87\,x_{\text{P64}} \\
&+ 40.70\,x_{\text{P65}} + 8.34\,x_{\text{P66}} + 26.11\,x_{\text{P67}} + 39.54\,x_{\text{P68}} + 14.07\,x_{\text{P69}} + 35.79\,x_{\text{P70}} + 6.92\,x_{\text{P71}} + 23.03\,x_{\text{P72}} \\
&+ 45.56\,x_{\text{P73}} + 17.20\,x_{\text{P74}} + 43.27\,x_{\text{P75}} + 9.52\,x_{\text{P76}} + 28.08\,x_{\text{P77}} + 41.10\,x_{\text{P78}} + 15.46\,x_{\text{P79}} + 34.30\,x_{\text{P80}} \\
&+ 5.43\,x_{\text{P81}} + 20.25\,x_{\text{P82}} + 48.38\,x_{\text{P83}} + 18.68\,x_{\text{P84}} + 40.01\,x_{\text{P85}} + 11.45\,x_{\text{P86}} + 25.32\,x_{\text{P87}} + 37.58\,x_{\text{P88}} \\
&+ 13.62\,x_{\text{P89}} + 31.84\,x_{\text{P90}} + 7.27\,x_{\text{P91}} + 24.75\,x_{\text{P92}} + 49.88\,x_{\text{P93}} + 16.85\,x_{\text{P94}} + 42.82\,x_{\text{P95}} + 10.13\,x_{\text{P96}} \\
&+ 27.36\,x_{\text{P97}} + 36.19\,x_{\text{P98}} + 12.80\,x_{\text{P99}} + 30.09\,x_{\text{P100}} + 6.07\,x_{\text{P101}} + 22.67\,x_{\text{P102}} + 44.42\,x_{\text{P103}} + 19.54\,x_{\text{P104}} \\
&+ 47.41\,x_{\text{P105}} + 8.78\,x_{\text{P106}} + 29.97\,x_{\text{P107}} + 38.96\,x_{\text{P108}} + 15.11\,x_{\text{P109}} + 33.05\,x_{\text{P110}} + 9.99\,x_{\text{P111}}
\end{aligned}
\right)
$$

**Subject to, for each device:**

- Device A:
$$
8.1\,x_{\text{P1}} + 2.5\,x_{\text{P2}} + 10.2\,x_{\text{P3}} + \cdots + 8.5\,x_{\text{P110}} + 8.1\,x_{\text{P111}} \leq 3500
$$

- Device B:
$$
10.5\,x_{\text{P1}} + 5.2\,x_{\text{P2}} + 8.3\,x_{\text{P3}} + \cdots + 2.7\,x_{\text{P110}} + 8.7\,x_{\text{P111}} \leq 4200
$$

- Device C:
$$
2.1\,x_{\text{P1}} + 13.4\,x_{\text{P2}} + 10.3\,x_{\text{P3}} + \cdots + 2.9\,x_{\text{P110}} + 2.9\,x_{\text{P111}} \leq 4500
$$

- Device D:
$$
5.8\,x_{\text{P1}} + 1.2\,x_{\text{P2}} + 11.7\,x_{\text{P3}} + \cdots + 2.0\,x_{\text{P110}} + 8.3\,x_{\text{P111}} \leq 2800
$$

- Device E:
$$
9.3\,x_{\text{P1}} + 4.1\,x_{\text{P2}} + 12.8\,x_{\text{P3}} + \cdots + 14.2\,x_{\text{P110}} + 6.7\,x_{\text{P111}} \leq 3300
$$

- Device F:
$$
3.8\,x_{\text{P1}} + 14.2\,x_{\text{P2}} + 9.7\,x_{\text{P3}} + \cdots + 2.4\,x_{\text{P110}} + 8.1\,x_{\text{P111}} \leq 3800
$$

- Device G:
$$
7.2\,x_{\text{P1}} + 2.8\,x_{\text{P2}} + 13.1\,x_{\text{P3}} + \cdots + 3.1\,x_{\text{P110}} + 5.9\,x_{\text{P111}} \leq 4100
$$

- Device H:
$$
11.7\,x_{\text{P1}} + 6.3\,x_{\text{P2}} + 1.9\,x_{\text{P3}} + \cdots + 7.8\,x_{\text{P110}} + 1.4\,x_{\text{P111}} \leq 3900
$$

- Device I:
$$
1.1\,x_{\text{P1}} + 11.3\,x_{\text{P2}} + 6.9\,x_{\text{P3}} + \cdots + 9.6\,x_{\text{P110}} + 7.3\,x_{\text{P111}} \leq 4800
$$

- Device J:
$$
4.6\,x_{\text{P1}} + 0.2\,x_{\text{P2}} + 10.4\,x_{\text{P3}} + \cdots + 11.4\,x_{\text{P110}} + 12.4\,x_{\text{P111}} \leq 3100
$$

(Where each "..." stands for the remaining coefficients from device_time.csv, in the original column order.)

**And for all $k$:**
$$
x_k \geq 0
$$

---

**Summary:**  
Maximize total profit from producing products P1–P111, subject to device time constraints for each device A–J, using the exact coefficients and capacities as provided. All variables are nonnegative and continuous.