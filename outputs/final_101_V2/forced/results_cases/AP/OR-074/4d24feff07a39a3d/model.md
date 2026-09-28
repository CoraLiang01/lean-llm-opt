##### Decision Variables:

Let $x_t$ be the number of waitstaff who start their 8-hour shift at time interval $t$, for $t = 1, 2, ..., 48$, where each $t$ corresponds to the time intervals as listed below.

##### Parameters:

Let $r_t$ be the required minimum number of waitstaff during time interval $t$, as given in the table below.

| $t$ | Time Interval            | $r_t$ |
|-----|-------------------------|-------|
| 1   | 2:00am - 2:30am         | 2     |
| 2   | 2:30am - 3:00am         | 3     |
| 3   | 3:00am - 3:30am         | 4     |
| 4   | 3:30am - 4:00am         | 6     |
| 5   | 4:00am - 4:30am         | 5     |
| 6   | 4:30am - 5:00am         | 4     |
| 7   | 5:00am - 5:30am         | 5     |
| 8   | 5:30am - 6:00am         | 6     |
| 9   | 6:00am - 6:30am         | 7     |
| 10  | 6:30am - 7:00am         | 8     |
| 11  | 7:00am - 7:30am         | 9     |
| 12  | 7:30am - 8:00am         | 9     |
| 13  | 8:00am - 8:30am         | 8     |
| 14  | 8:30am - 9:00am         | 8     |
| 15  | 9:00am - 9:30am         | 9     |
| 16  | 9:30am - 10:00am        | 9     |
| 17  | 10:00am - 10:30am       | 10    |
| 18  | 10:30am - 11:00am       | 12    |
| 19  | 11:00am - 11:30am       | 11    |
| 20  | 11:30am - 12:00pm       | 11    |
| 21  | 12:00pm - 12:30pm       | 12    |
| 22  | 12:30pm - 1:00pm        | 11    |
| 23  | 1:00pm - 1:30pm         | 10    |
| 24  | 1:30pm - 2:00pm         | 9     |
| 25  | 2:00pm - 2:30pm         | 8     |
| 26  | 2:30pm - 3:00pm         | 7     |
| 27  | 3:00pm - 3:30pm         | 6     |
| 28  | 3:30pm - 4:00pm         | 5     |
| 29  | 4:00pm - 4:30pm         | 5     |
| 30  | 4:30pm - 5:00pm         | 6     |
| 31  | 5:00pm - 5:30pm         | 7     |
| 32  | 5:30pm - 6:00pm         | 8     |
| 33  | 6:00pm - 6:30pm         | 9     |
| 34  | 6:30pm - 7:00pm         | 10    |
| 35  | 7:00pm - 7:30pm         | 9     |
| 36  | 7:30pm - 8:00pm         | 8     |
| 37  | 8:00pm - 8:30pm         | 7     |
| 38  | 8:30pm - 9:00pm         | 6     |
| 39  | 9:00pm - 9:30pm         | 5     |
| 40  | 9:30pm - 10:00pm        | 4     |
| 41  | 10:00pm - 10:30pm       | 4     |
| 42  | 10:30pm - 11:00pm       | 3     |
| 43  | 11:00pm - 11:30pm       | 3     |
| 44  | 11:30pm - 12:00am       | 3     |
| 45  | 12:00am - 12:30am       | 3     |
| 46  | 12:30am - 1:00am        | 4     |
| 47  | 1:00am - 1:30am         | 4     |
| 48  | 1:30am - 2:00am         | 4     |

##### Objective Function:

$\min \sum_{t=1}^{48} x_t$

##### Constraints:

For each time interval $s = 1, 2, ..., 48$:

$\sum_{k=0}^{15} x_{(s - k - 1) \bmod 48 + 1} \geq r_s$

where $x_{(s - k - 1) \bmod 48 + 1}$ represents the number of staff who started in the 8 hours (16 half-hour intervals) prior to and including interval $s$ (since each shift covers 16 consecutive intervals).

##### Variable Constraints:

$x_t \geq 0$ and integer, for all $t = 1, ..., 48$

##### Retrieved Information

{
  "requirement": [
    {"interval": "2:00am - 2:30am", "value": 2},
    {"interval": "2:30am - 3:00am", "value": 3},
    {"interval": "3:00am - 3:30am", "value": 4},
    {"interval": "3:30am - 4:00am", "value": 6},
    {"interval": "4:00am - 4:30am", "value": 5},
    {"interval": "4:30am - 5:00am", "value": 4},
    {"interval": "5:00am - 5:30am", "value": 5},
    {"interval": "5:30am - 6:00am", "value": 6},
    {"interval": "6:00am - 6:30am", "value": 7},
    {"interval": "6:30am - 7:00am", "value": 8},
    {"interval": "7:00am - 7:30am", "value": 9},
    {"interval": "7:30am - 8:00am", "value": 9},
    {"interval": "8:00am - 8:30am", "value": 8},
    {"interval": "8:30am - 9:00am", "value": 8},
    {"interval": "9:00am - 9:30am", "value": 9},
    {"interval": "9:30am - 10:00am", "value": 9},
    {"interval": "10:00am - 10:30am", "value": 10},
    {"interval": "10:30am - 11:00am", "value": 12},
    {"interval": "11:00am - 11:30am", "value": 11},
    {"interval": "11:30am - 12:00pm", "value": 11},
    {"interval": "12:00pm - 12:30pm", "value": 12},
    {"interval": "12:30pm - 1:00pm", "value": 11},
    {"interval": "1:00pm - 1:30pm", "value": 10},
    {"interval": "1:30pm - 2:00pm", "value": 9},
    {"interval": "2:00pm - 2:30pm", "value": 8},
    {"interval": "2:30pm - 3:00pm", "value": 7},
    {"interval": "3:00pm - 3:30pm", "value": 6},
    {"interval": "3:30pm - 4:00pm", "value": 5},
    {"interval": "4:00pm - 4:30pm", "value": 5},
    {"interval": "4:30pm - 5:00pm", "value": 6},
    {"interval": "5:00pm - 5:30pm", "value": 7},
    {"interval": "5:30pm - 6:00pm", "value": 8},
    {"interval": "6:00pm - 6:30pm", "value": 9},
    {"interval": "6:30pm - 7:00pm", "value": 10},
    {"interval": "7:00pm - 7:30pm", "value": 9},
    {"interval": "7:30pm - 8:00pm", "value": 8},
    {"interval": "8:00pm - 8:30pm", "value": 7},
    {"interval": "8:30pm - 9:00pm", "value": 6},
    {"interval": "9:00pm - 9:30pm", "value": 5},
    {"interval": "9:30pm - 10:00pm", "value": 4},
    {"interval": "10:00pm - 10:30pm", "value": 4},
    {"interval": "10:30pm - 11:00pm", "value": 3},
    {"interval": "11:00pm - 11:30pm", "value": 3},
    {"interval": "11:30pm - 12:00am", "value": 3},
    {"interval": "12:00am - 12:30am", "value": 3},
    {"interval": "12:30am - 1:00am", "value": 4},
    {"interval": "1:00am - 1:30am", "value": 4},
    {"interval": "1:30am - 2:00am", "value": 4}
  ],
  "intervals": [
    "2:00am - 2:30am",
    "2:30am - 3:00am",
    "3:00am - 3:30am",
    "3:30am - 4:00am",
    "4:00am - 4:30am",
    "4:30am - 5:00am",
    "5:00am - 5:30am",
    "5:30am - 6:00am",
    "6:00am - 6:30am",
    "6:30am - 7:00am",
    "7:00am - 7:30am",
    "7:30am - 8:00am",
    "8:00am - 8:30am",
    "8:30am - 9:00am",
    "9:00am - 9:30am",
    "9:30am - 10:00am",
    "10:00am - 10:30am",
    "10:30am - 11:00am",
    "11:00am - 11:30am",
    "11:30am - 12:00pm",
    "12:00pm - 12:30pm",
    "12:30pm - 1:00pm",
    "1:00pm - 1:30pm",
    "1:30pm - 2:00pm",
    "2:00pm - 2:30pm",
    "2:30pm - 3:00pm",
    "3:00pm - 3:30pm",
    "3:30pm - 4:00pm",
    "4:00pm - 4:30pm",
    "4:30pm - 5:00pm",
    "5:00pm - 5:30pm",
    "5:30pm - 6:00pm",
    "6:00pm - 6:30pm",
    "6:30pm - 7:00pm",
    "7:00pm - 7:30pm",
    "7:30pm - 8:00pm",
    "8:00pm - 8:30pm",
    "8:30pm - 9:00pm",
    "9:00pm - 9:30pm",
    "9:30pm - 10:00pm",
    "10:00pm - 10:30pm",
    "10:30pm - 11:00pm",
    "11:00pm - 11:30pm",
    "11:30pm - 12:00am",
    "12:00am - 12:30am",
    "12:30am - 1:00am",
    "1:00am - 1:30am",
    "1:30am - 2:00am"
  ]
}