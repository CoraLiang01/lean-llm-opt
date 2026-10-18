Let the day be divided into $N=48$ half-hour periods, indexed by $t=1,2,\ldots,48$, in source order as in the data below. Let $r_t$ be the required minimum number of waitstaff in period $t$. Let $x_t\geq0$ (integer) be the number of waitstaff whose shift starts at period $t$. Each shift lasts for 8 hours = 16 periods.

Objective:
$$
\min \sum_{t=1}^{48} x_t
$$

Constraints:
For each period $t=1,\ldots,48$,
$$
\sum_{k=0}^{15} x_{(t-k-1 \bmod 48)+1} \geq r_t
$$
where the sum is over the 16 periods whose shifts cover period $t$ (including those that started up to 7.5 hours earlier, wrapping around midnight).

Variable domains:
$$
x_t \in \mathbb{Z}_{\geq 0},\quad t=1,\ldots,48
$$

Data (in source order):

| $t$ | Time                     | $r_t$ |
|----:|-------------------------|------:|
|  1  | 2:00am - 2:30am         |   2   |
|  2  | 2:30am - 3:00am         |   3   |
|  3  | 3:00am - 3:30am         |   4   |
|  4  | 3:30am - 4:00am         |   6   |
|  5  | 4:00am - 4:30am         |   5   |
|  6  | 4:30am - 5:00am         |   4   |
|  7  | 5:00am - 5:30am         |   5   |
|  8  | 5:30am - 6:00am         |   6   |
|  9  | 6:00am - 6:30am         |   7   |
| 10  | 6:30am - 7:00am         |   8   |
| 11  | 7:00am - 7:30am         |   9   |
| 12  | 7:30am - 8:00am         |   9   |
| 13  | 8:00am - 8:30am         |   8   |
| 14  | 8:30am - 9:00am         |   8   |
| 15  | 9:00am - 9:30am         |   9   |
| 16  | 9:30am - 10:00am        |   9   |
| 17  | 10:00am - 10:30am       |  10   |
| 18  | 10:30am - 11:00am       |  12   |
| 19  | 11:00am - 11:30am       |  11   |
| 20  | 11:30am - 12:00pm       |  11   |
| 21  | 12:00pm - 12:30pm       |  12   |
| 22  | 12:30pm - 1:00pm        |  11   |
| 23  | 1:00pm - 1:30pm         |  10   |
| 24  | 1:30pm - 2:00pm         |   9   |
| 25  | 2:00pm - 2:30pm         |   8   |
| 26  | 2:30pm - 3:00pm         |   7   |
| 27  | 3:00pm - 3:30pm         |   6   |
| 28  | 3:30pm - 4:00pm         |   5   |
| 29  | 4:00pm - 4:30pm         |   5   |
| 30  | 4:30pm - 5:00pm         |   6   |
| 31  | 5:00pm - 5:30pm         |   7   |
| 32  | 5:30pm - 6:00pm         |   8   |
| 33  | 6:00pm - 6:30pm         |   9   |
| 34  | 6:30pm - 7:00pm         |  10   |
| 35  | 7:00pm - 7:30pm         |   9   |
| 36  | 7:30pm - 8:00pm         |   8   |
| 37  | 8:00pm - 8:30pm         |   7   |
| 38  | 8:30pm - 9:00pm         |   6   |
| 39  | 9:00pm - 9:30pm         |   5   |
| 40  | 9:30pm - 10:00pm        |   4   |
| 41  | 10:00pm - 10:30pm       |   4   |
| 42  | 10:30pm - 11:00pm       |   3   |
| 43  | 11:00pm - 11:30pm       |   3   |
| 44  | 11:30pm - 12:00am       |   3   |
| 45  | 12:00am - 12:30am       |   3   |
| 46  | 12:30am - 1:00am        |   4   |
| 47  | 1:00am - 1:30am         |   4   |
| 48  | 1:30am - 2:00am         |   4   |

All requirements and time periods are as in the table above, in source order.