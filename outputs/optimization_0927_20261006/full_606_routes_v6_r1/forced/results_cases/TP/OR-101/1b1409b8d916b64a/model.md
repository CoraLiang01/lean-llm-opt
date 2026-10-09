Let $x_p \geq 0$ be the continuous production quantity of product $p \in \{P1, P2, \ldots, P111\}$.

Maximize total monthly profit:
$$
\max \sum_{p=1}^{111} \pi_p x_p
$$
where $\pi_p$ is the unit profit of product $p$ (see table below).

Subject to device capacity constraints for each device $d \in \{A, B, C, D, E, F, G, H, I, J\}$:
$$
\sum_{p=1}^{111} t_{d,p} x_p \leq C_d \qquad \forall d \in \{A, B, C, D, E, F, G, H, I, J\}
$$
where $t_{d,p}$ is the processing time required by product $p$ on device $d$, and $C_d$ is the monthly capacity of device $d$.

And nonnegativity:
$$
x_p \geq 0 \qquad \forall p \in \{P1, P2, \ldots, P111\}
$$

---

#### Data

**Products:** $P1$ through $P111$

**Devices:** $A$, $B$, $C$, $D$, $E$, $F$, $G$, $H$, $I$, $J$

**Unit Profits $\pi_p$ (by product, in source order):**

| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| P3      | 45.21       |
| P4      | 18.92       |
| P5      | 33.47       |
| P6      | 8.64        |
| P7      | 25.88       |
| P8      | 40.15       |
| P9      | 14.39       |
| P10     | 37.62       |
| P11     | 10.25       |
| P12     | 29.81       |
| P13     | 48.77       |
| P14     | 16.53       |
| P15     | 42.09       |
| P16     | 9.11        |
| P17     | 22.46       |
| P18     | 36.73       |
| P19     | 11.99       |
| P20     | 31.28       |
| P21     | 7.45        |
| P22     | 27.6        |
| P23     | 43.91       |
| P24     | 15.82       |
| P25     | 39.36       |
| P26     | 6.23        |
| P27     | 20.71       |
| P28     | 34.98       |
| P29     | 13.24       |
| P30     | 32.53       |
| P31     | 5.1         |
| P32     | 24.18       |
| P33     | 47.85       |
| P34     | 19.3        |
| P35     | 41.44       |
| P36     | 11.02       |
| P37     | 26.57       |
| P38     | 38.2        |
| P39     | 17.76       |
| P40     | 46.12       |
| P41     | 9.89        |
| P42     | 21.34       |
| P43     | 33.86       |
| P44     | 14.88       |
| P45     | 30.41       |
| P46     | 6.5         |
| P47     | 28.93       |
| P48     | 49.2        |
| P49     | 18.15       |
| P50     | 44.78       |
| P51     | 10.57       |
| P52     | 23.69       |
| P53     | 35.51       |
| P54     | 16.03       |
| P55     | 38.83       |
| P56     | 5.88        |
| P57     | 29.45       |
| P58     | 42.33       |
| P59     | 12.41       |
| P60     | 37.06       |
| P61     | 7.99        |
| P62     | 21.9        |
| P63     | 46.99       |
| P64     | 19.87       |
| P65     | 40.7        |
| P66     | 8.34        |
| P67     | 26.11       |
| P68     | 39.54       |
| P69     | 14.07       |
| P70     | 35.79       |
| P71     | 6.92        |
| P72     | 23.03       |
| P73     | 45.56       |
| P74     | 17.2        |
| P75     | 43.27       |
| P76     | 9.52        |
| P77     | 28.08       |
| P78     | 41.1        |
| P79     | 15.46       |
| P80     | 34.3        |
| P81     | 5.43        |
| P82     | 20.25       |
| P83     | 48.38       |
| P84     | 18.68       |
| P85     | 40.01       |
| P86     | 11.45       |
| P87     | 25.32       |
| P88     | 37.58       |
| P89     | 13.62       |
| P90     | 31.84       |
| P91     | 7.27        |
| P92     | 24.75       |
| P93     | 49.88       |
| P94     | 16.85       |
| P95     | 42.82       |
| P96     | 10.13       |
| P97     | 27.36       |
| P98     | 36.19       |
| P99     | 12.8        |
| P100    | 30.09       |
| P101    | 6.07        |
| P102    | 22.67       |
| P103    | 44.42       |
| P104    | 19.54       |
| P105    | 47.41       |
| P106    | 8.78        |
| P107    | 29.97       |
| P108    | 38.96       |
| P109    | 15.11       |
| P110    | 33.05       |
| P111    | 9.99        |

**Device Monthly Capacities $C_d$:**

| Device | Monthly_Capacity |
|--------|------------------|
| A      | 3500             |
| B      | 4200             |
| C      | 4500             |
| D      | 2800             |
| E      | 3300             |
| F      | 3800             |
| G      | 4100             |
| H      | 3900             |
| I      | 4800             |
| J      | 3100             |

**Processing Times $t_{d,p}$ (device $d$ by row, product $p$ by column, in source order):**

- For each device $d \in \{A, B, C, D, E, F, G, H, I, J\}$, the processing time for product $p$ is as given in the corresponding row of device_time.csv (see above for full data).

---

**Summary of Model:**

- Variables: $x_p \geq 0$ for $p = P1, \ldots, P111$
- Objective: $\max \sum_{p=1}^{111} \pi_p x_p$
- Constraints: For each device $d$, $\sum_{p=1}^{111} t_{d,p} x_p \leq C_d$
- All coefficients and identifiers as retrieved above.