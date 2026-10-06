Facility Costs (facility_costs.csv):

| Facility | FixedCost | Capacity |
|----------|-----------|----------|
| A1       | 0         | 30       |  (row 1)
| A2       | 175       | 10       |  (row 2)
| A3       | 300       | 20       |  (row 3)
| A4       | 375       | 30       |  (row 4)
| A5       | 500       | 40       |  (row 5)
| A6       | 200       | 20       |  (row 6)
| A7       | 260       | 25       |  (row 7)
| A8       | 220       | 30       |  (row 8)
| A9       | 320       | 35       |  (row 9)
| A10      | 280       | 20       |  (row 10)
| A11      | 350       | 40       |  (row 11)
| A12      | 420       | 25       |  (row 12)
| A13      | 470       | 30       |  (row 13)
| A14      | 520       | 50       |  (row 14)
| A15      | 560       | 45       |  (row 15)

Demand Requirements (demand_requirements.csv):

| Destination | Demand |
|-------------|--------|
| B1          | 30     |  (row 1)
| B2          | 25     |  (row 2)
| B3          | 20     |  (row 3)
| B4          | 35     |  (row 4)
| B5          | 25     |  (row 5)
| B6          | 30     |  (row 6)
| B7          | 25     |  (row 7)
| B8          | 30     |  (row 8)

Shipping Costs (shipping_costs.csv):

| Origin | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|--------|----|----|----|----|----|----|----|----|
| A1     | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |  (row 1)
| A2     | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |  (row 2)
| A3     | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |  (row 3)
| A4     | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |  (row 4)
| A5     | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |  (row 5)
| A6     | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |  (row 6)
| A7     | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |  (row 7)
| A8     | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |  (row 8)
| A9     | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |  (row 9)
| A10    | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |  (row 10)
| A11    | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |  (row 11)
| A12    | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |  (row 12)
| A13    | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |  (row 13)
| A14    | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |  (row 14)
| A15    | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |  (row 15)

All identifiers (Facility, Origin, Destination, Customer) and values are preserved in their original source order and orientation. No data has been omitted, transposed, or inferred.