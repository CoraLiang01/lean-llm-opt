##### Decision Variables

$y_t\geq0$: number of waitstaff whose shift starts at the beginning of period $t$, for $t=1,\dots ,48$ (continuous).

##### Objective Function

$\displaystyle \min \sum_{t=1}^{48} y_t$

##### Constraints

1. **Coverage of each half‑hour period**  
   For every period $p=1,\dots ,48$ the total number of staff working during that period must be at least the required number of waitstaff $r_p$:

   $\displaystyle \sum_{\substack{t=1 \\ t\le p \le t+15}}^{48} y_t \;\ge\; r_p ,\qquad \forall\, p=1,\dots ,48$

2. **Non‑negativity**  
   $\displaystyle y_t \;\ge\; 0 ,\qquad \forall\, t=1,\dots ,48$

##### Period Indexing

| $t$ | Time Interval |
|-----|---------------|
| 1 | 2:00 am – 2:30 am |
| 2 | 2:30 am – 3:00 am |
| 3 | 3:00 am – 3:30 am |
| 4 | 3:30 am – 4:00 am |
| 5 | 4:00 am – 4:30 am |
| 6 | 4:30 am – 5:00 am |
| 7 | 5:00 am – 5:30 am |
| 8 | 5:30 am – 6:00 am |
| 9 | 6:00 am – 6:30 am |
|10 | 6:30 am – 7:00 am |
|11 | 7:00 am – 7:30 am |
|12 | 7:30 am – 8:00 am |
|13 | 8:00 am – 8:30 am |
|14 | 8:30 am – 9:00 am |
|15 | 9:00 am – 9:30 am |
|16 | 9:30 am – 10:00 am |
|17 | 10:00 am – 10:30 am |
|18 | 10:30 am – 11:00 am |
|19 | 11:00 am – 11:30 am |
|20 | 11:30 am – 12:00 pm |
|21 | 12:00 pm – 12:30 pm |
|22 | 12:30 pm – 1:00 pm |
|23 | 1:00 pm – 1:30 pm |
|24 | 1:30 pm – 2:00 pm |
|25 | 2:00 pm – 2:30 pm |
|26 | 2:30 pm – 3:00 pm |
|27 | 3:00 pm – 3:30 pm |
|28 | 3:30 pm – 4:00 pm |
|29 | 4:00 pm – 4:30 pm |
|30 | 4:30 pm – 5:00 pm |
|31 | 5:00 pm – 5:30 pm |
|32 | 5:30 pm – 6:00 pm |
|33 | 6:00 pm – 6:30 pm |
|34 | 6:30 pm – 7:00 pm |
|35 | 7:00 pm – 7:30 pm |
|36 | 7:30 pm – 8:00 pm |
|37 | 8:00 pm – 8:30 pm |
|38 | 8:30 pm – 9:00 pm |
|39 | 9:00 pm – 9:30 pm |
|40 | 9:30 pm – 10:00 pm |
|41 | 10:00 pm – 10:30 pm |
|42 | 10:30 pm – 11:00 pm |
|43 | 11:00 pm – 11:30 pm |
|44 | 11:30 pm – 12:00 am |
|45 | 12:00 am – 12:30 am |
|46 | 12:30 am – 1:00 am |
|47 | 1:00 am – 1:30 am |
|48 | 1:30 am – 2:00 am |

##### Requirement Data $r_p$

| $p$ | $r_p$ |
|-----|-------|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 6 |
| 5 | 5 |
| 6 | 4 |
| 7 | 5 |
| 8 | 6 |
| 9 | 7 |
|10 | 8 |
|11 | 9 |
|12 | 9 |
|13 | 8 |
|14 | 8 |
|15 | 9 |
|16 | 9 |
|17 |10 |
|18 |12 |
|19 |11 |
|20 |11 |
|21 |12 |
|22 |11 |
|23 |10 |
|24 | 9 |
|25 | 8 |
|26 | 7 |
|27 | 6 |
|28 | 5 |
|29 | 5 |
|30 | 6 |
|31 | 7 |
|32 | 8 |
|33 | 9 |
|34 |10 |
|35 | 9 |
|36 | 8 |
|37 | 7 |
|38 | 6 |
|39 | 5 |
|40 | 4 |
|41 | 4 |
|42 | 3 |
|43 | 3 |
|44 | 3 |
|45 | 3 |
|46 | 4 |
|47 | 4 |
|48 | 4 |

##### Retrieved Information

The data above were extracted from **44.csv** and provide the full set of hourly (half‑hourly) staffing requirements needed to formulate the linear programming model.