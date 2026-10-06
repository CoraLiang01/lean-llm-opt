Minimize
    obj: 11 y_B1 + 14 y_B2 + 10 y_B3 + 13 y_B4 + 16 y_B5 + 9 y_B6 + 12 y_B7 + 15 y_B8
    Subject To
    cover_Z1: y_B1 + y_B4 >= 1
    cover_Z2: y_B1 + y_B2 >= 1
    cover_Z3: y_B2 + y_B5 >= 1
    cover_Z4: y_B3 + y_B7 >= 1
    cover_Z5: y_B1 + y_B3 + y_B8 >= 1
    cover_Z6: y_B2 + y_B4 + y_B8 >= 1
    cover_Z7: y_B4 + y_B5 >= 1
    cover_Z8: y_B3 + y_B6 >= 1
    cover_Z9: y_B5 + y_B6 + y_B8 >= 1
    cover_Z10: y_B6 + y_B7 >= 1
    Binaries
    y_B1 y_B2 y_B3 y_B4 y_B5 y_B6 y_B7 y_B8
    End
