Let $x_t$ be the number of waitstaff who start their 8-hour shift at time period $t$, for $t=1,\ldots,48$, where each period is a half-hour interval as listed below.

Minimize the total number of waitstaff scheduled:
$$
\min \sum_{t=1}^{48} x_t
$$

Subject to, for each time period $k=1,\ldots,48$:
$$
\sum_{i=0}^{15} x_{(k-i-1 \bmod 48)+1} \geq r_k
$$

where $r_k$ is the required minimum number of waitstaff for period $k$ (see table below), and $x_t \geq 0$ and integer for all $t$.

Here, each $x_t$ covers periods $t, t+1, \ldots, t+15$ (modulo 48), since each shift lasts 8 hours (16 half-hour periods).

#### Periods and Requirements (in source order):

| $t$ | Time                   | $r_t$ |
|-----|------------------------|-------|
| 1   | 2:00am - 2:30am        | 2     |
| 2   | 2:30am - 3:00am        | 3     |
| 3   | 3:00am - 3:30am        | 4     |
| 4   | 3:30am - 4:00am        | 6     |
| 5   | 4:00am - 4:30am        | 5     |
| 6   | 4:30am - 5:00am        | 4     |
| 7   | 5:00am - 5:30am        | 5     |
| 8   | 5:30am - 6:00am        | 6     |
| 9   | 6:00am - 6:30am        | 7     |
| 10  | 6:30am - 7:00am        | 8     |
| 11  | 7:00am - 7:30am        | 9     |
| 12  | 7:30am - 8:00am        | 9     |
| 13  | 8:00am - 8:30am        | 8     |
| 14  | 8:30am - 9:00am        | 8     |
| 15  | 9:00am - 9:30am        | 9     |
| 16  | 9:30am - 10:00am       | 9     |
| 17  | 10:00am - 10:30am      | 10    |
| 18  | 10:30am - 11:00am      | 12    |
| 19  | 11:00am - 11:30am      | 11    |
| 20  | 11:30am - 12:00pm      | 11    |
| 21  | 12:00pm - 12:30pm      | 12    |
| 22  | 12:30pm - 1:00pm       | 11    |
| 23  | 1:00pm - 1:30pm        | 10    |
| 24  | 1:30pm - 2:00pm        | 9     |
| 25  | 2:00pm - 2:30pm        | 8     |
| 26  | 2:30pm - 3:00pm        | 7     |
| 27  | 3:00pm - 3:30pm        | 6     |
| 28  | 3:30pm - 4:00pm        | 5     |
| 29  | 4:00pm - 4:30pm        | 5     |
| 30  | 4:30pm - 5:00pm        | 6     |
| 31  | 5:00pm - 5:30pm        | 7     |
| 32  | 5:30pm - 6:00pm        | 8     |
| 33  | 6:00pm - 6:30pm        | 9     |
| 34  | 6:30pm - 7:00pm        | 10    |
| 35  | 7:00pm - 7:30pm        | 9     |
| 36  | 7:30pm - 8:00pm        | 8     |
| 37  | 8:00pm - 8:30pm        | 7     |
| 38  | 8:30pm - 9:00pm        | 6     |
| 39  | 9:00pm - 9:30pm        | 5     |
| 40  | 9:30pm - 10:00pm       | 4     |
| 41  | 10:00pm - 10:30pm      | 4     |
| 42  | 10:30pm - 11:00pm      | 3     |
| 43  | 11:00pm - 11:30pm      | 3     |
| 44  | 11:30pm - 12:00am      | 3     |
| 45  | 12:00am - 12:30am      | 3     |
| 46  | 12:30am - 1:00am       | 4     |
| 47  | 1:00am - 1:30am        | 4     |
| 48  | 1:30am - 2:00am        | 4     |

Variables:
$$
x_t \in \mathbb{Z}_{\geq 0},\quad t=1,\ldots,48
$$

where $x_t$ is the number of waitstaff starting at the beginning of period $t$.

All requirements and time periods are as listed above, in the same order as in the source data.