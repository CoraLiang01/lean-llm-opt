Let $x_p \geq 0$ denote the production quantity (in continuous units) of product $p \in \{P1, P2, \ldots, P111\}$.

**Parameters:**

- $c_p$: Unit profit of product $p$ (from unit_product_profits.csv)
- $a_{dp}$: Processing time required by product $p$ on device $d$ (from device_time.csv)
- $b_d$: Monthly capacity of device $d$ (from monthly_device_capacity.csv)

**Objective:**

$$
\max \sum_{p \in \{P1,\ldots,P111\}} c_p \, x_p
$$

where the $c_p$ values are:

- $c_{P1} = 28.55$
- $c_{P2} = 12.78$
- $c_{P3} = 45.21$
- $c_{P4} = 18.92$
- $c_{P5} = 33.47$
- $c_{P6} = 8.64$
- $c_{P7} = 25.88$
- $c_{P8} = 40.15$
- $c_{P9} = 14.39$
- $c_{P10} = 37.62$
- $c_{P11} = 10.25$
- $c_{P12} = 29.81$
- $c_{P13} = 48.77$
- $c_{P14} = 16.53$
- $c_{P15} = 42.09$
- $c_{P16} = 9.11$
- $c_{P17} = 22.46$
- $c_{P18} = 36.73$
- $c_{P19} = 11.99$
- $c_{P20} = 31.28$
- $c_{P21} = 7.45$
- $c_{P22} = 27.6$
- $c_{P23} = 43.91$
- $c_{P24} = 15.82$
- $c_{P25} = 39.36$
- $c_{P26} = 6.23$
- $c_{P27} = 20.71$
- $c_{P28} = 34.98$
- $c_{P29} = 13.24$
- $c_{P30} = 32.53$
- $c_{P31} = 5.1$
- $c_{P32} = 24.18$
- $c_{P33} = 47.85$
- $c_{P34} = 19.3$
- $c_{P35} = 41.44$
- $c_{P36} = 11.02$
- $c_{P37} = 26.57$
- $c_{P38} = 38.2$
- $c_{P39} = 17.76$
- $c_{P40} = 46.12$
- $c_{P41} = 9.89$
- $c_{P42} = 21.34$
- $c_{P43} = 33.86$
- $c_{P44} = 14.88$
- $c_{P45} = 30.41$
- $c_{P46} = 6.5$
- $c_{P47} = 28.93$
- $c_{P48} = 49.2$
- $c_{P49} = 18.15$
- $c_{P50} = 44.78$
- $c_{P51} = 10.57$
- $c_{P52} = 23.69$
- $c_{P53} = 35.51$
- $c_{P54} = 16.03$
- $c_{P55} = 38.83$
- $c_{P56} = 5.88$
- $c_{P57} = 29.45$
- $c_{P58} = 42.33$
- $c_{P59} = 12.41$
- $c_{P60} = 37.06$
- $c_{P61} = 7.99$
- $c_{P62} = 21.9$
- $c_{P63} = 46.99$
- $c_{P64} = 19.87$
- $c_{P65} = 40.7$
- $c_{P66} = 8.34$
- $c_{P67} = 26.11$
- $c_{P68} = 39.54$
- $c_{P69} = 14.07$
- $c_{P70} = 35.79$
- $c_{P71} = 6.92$
- $c_{P72} = 23.03$
- $c_{P73} = 45.56$
- $c_{P74} = 17.2$
- $c_{P75} = 43.27$
- $c_{P76} = 9.52$
- $c_{P77} = 28.08$
- $c_{P78} = 41.1$
- $c_{P79} = 15.46$
- $c_{P80} = 34.3$
- $c_{P81} = 5.43$
- $c_{P82} = 20.25$
- $c_{P83} = 48.38$
- $c_{P84} = 18.68$
- $c_{P85} = 40.01$
- $c_{P86} = 11.45$
- $c_{P87} = 25.32$
- $c_{P88} = 37.58$
- $c_{P89} = 13.62$
- $c_{P90} = 31.84$
- $c_{P91} = 7.27$
- $c_{P92} = 24.75$
- $c_{P93} = 49.88$
- $c_{P94} = 16.85$
- $c_{P95} = 42.82$
- $c_{P96} = 10.13$
- $c_{P97} = 27.36$
- $c_{P98} = 36.19$
- $c_{P99} = 12.8$
- $c_{P100} = 30.09$
- $c_{P101} = 6.07$
- $c_{P102} = 22.67$
- $c_{P103} = 44.42$
- $c_{P104} = 19.54$
- $c_{P105} = 47.41$
- $c_{P106} = 8.78$
- $c_{P107} = 29.97$
- $c_{P108} = 38.96$
- $c_{P109} = 15.11$
- $c_{P110} = 33.05$
- $c_{P111} = 9.99$

**Constraints:**

For each device $d \in \{A, B, C, D, E, F, G, H, I, J\}$,

$$
\sum_{p \in \{P1,\ldots,P111\}} a_{dp} \, x_p \leq b_d
$$

where the $a_{dp}$ values are as given in device_time.csv, i.e., for each device $d$ (A through J), and each product $p$ (P1 through P111), $a_{dp}$ is the processing time required by product $p$ on device $d$ (see the CSV data above for all coefficients).

The $b_d$ values (monthly device capacities) are:

- $b_A = 3500$
- $b_B = 4200$
- $b_C = 4500$
- $b_D = 2800$
- $b_E = 3300$
- $b_F = 3800$
- $b_G = 4100$
- $b_H = 3900$
- $b_I = 4800$
- $b_J = 3100$

**Variable domains:**

$$
x_p \geq 0 \quad \forall p \in \{P1, \ldots, P111\}
$$

**Summary of Model:**

Maximize
$$
\sum_{p = P1}^{P111} c_p \, x_p
$$

Subject to, for each device $d \in \{A, B, C, D, E, F, G, H, I, J\}$:
$$
\sum_{p = P1}^{P111} a_{dp} \, x_p \leq b_d
$$

and
$$
x_p \geq 0 \quad \forall p \in \{P1, \ldots, P111\}
$$

All coefficients and identifiers are as given in the retrieved CSV data above.