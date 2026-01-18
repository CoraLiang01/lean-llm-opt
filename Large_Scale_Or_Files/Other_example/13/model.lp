\ Model TwoFactoryProduction
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x44ad37d4e60d5125
Maximize
  12 x_A_P1 + 16 x_A_P2 + 10 x_A_P3 + 18 x_A_P4 + 10 x_A_P5 + 12 x_A_P6
   + 14 x_A_P7 + 16 x_A_P8 + 18 x_A_P9 + 10 x_A_P10 + 12 x_A_P11
   + 14 x_A_P12 + 16 x_A_P13 + 18 x_A_P14 + 10 x_A_P15 + 12 x_A_P16
   + 14 x_A_P17 + 16 x_A_P18 + 18 x_A_P19 + 10 x_A_P20 + 12 x_A_P21
   + 14 x_A_P22 + 16 x_A_P23 + 18 x_A_P24 + 10 x_A_P25 + 12 x_A_P26
   + 14 x_A_P27 + 16 x_A_P28 + 18 x_A_P29 + 10 x_A_P30 + 12 x_A_P31
   + 14 x_A_P32 + 16 x_A_P33 + 18 x_A_P34 + 10 x_A_P35 + 12 x_A_P36
   + 14 x_A_P37 + 16 x_A_P38 + 18 x_A_P39 + 10 x_A_P40 + 12 x_A_P41
   + 14 x_A_P42 + 16 x_A_P43 + 18 x_A_P44 + 10 x_A_P45 + 12 x_A_P46
   + 14 x_A_P47 + 16 x_A_P48 + 18 x_A_P49 + 10 x_A_P50 + 12 x_A_P51
   + 14 x_A_P52 + 16 x_A_P53 + 18 x_A_P54 + 10 x_A_P55 + 12 x_A_P56
   + 14 x_A_P57 + 16 x_A_P58 + 18 x_A_P59 + 10 x_A_P60 + 12 x_A_P61
   + 14 x_A_P62 + 15 x_B_P1 + 14 x_B_P2 + 11 x_B_P3 + 13 x_B_P4 + 16 x_B_P5
   + 9 x_B_P6 + 12 x_B_P7 + 15 x_B_P8 + 18 x_B_P9 + 11 x_B_P10 + 14 x_B_P11
   + 17 x_B_P12 + 10 x_B_P13 + 13 x_B_P14 + 16 x_B_P15 + 9 x_B_P16
   + 12 x_B_P17 + 15 x_B_P18 + 18 x_B_P19 + 11 x_B_P20 + 14 x_B_P21
   + 17 x_B_P22 + 10 x_B_P23 + 13 x_B_P24 + 16 x_B_P25 + 9 x_B_P26
   + 12 x_B_P27 + 15 x_B_P28 + 18 x_B_P29 + 11 x_B_P30 + 14 x_B_P31
   + 17 x_B_P32 + 10 x_B_P33 + 13 x_B_P34 + 16 x_B_P35 + 9 x_B_P36
   + 12 x_B_P37 + 15 x_B_P38 + 18 x_B_P39 + 11 x_B_P40 + 14 x_B_P41
   + 17 x_B_P42 + 10 x_B_P43 + 13 x_B_P44 + 16 x_B_P45 + 9 x_B_P46
   + 12 x_B_P47 + 15 x_B_P48 + 18 x_B_P49 + 11 x_B_P50 + 14 x_B_P51
   + 17 x_B_P52 + 10 x_B_P53 + 13 x_B_P54 + 16 x_B_P55 + 9 x_B_P56
   + 12 x_B_P57 + 15 x_B_P58 + 18 x_B_P59 + 11 x_B_P60 + 14 x_B_P61
   + 17 x_B_P62
Subject To
 cap_A_Labor_Hours: 2 x_A_P1 + 4 x_A_P2 + 3 x_A_P3 + x_A_P4 + 3 x_A_P5
   + x_A_P6 + 3 x_A_P7 + x_A_P8 + 3 x_A_P9 + x_A_P10 + 3 x_A_P11 + x_A_P12
   + 3 x_A_P13 + x_A_P14 + 3 x_A_P15 + x_A_P16 + 3 x_A_P17 + x_A_P18
   + 3 x_A_P19 + x_A_P20 + 3 x_A_P21 + x_A_P22 + 3 x_A_P23 + x_A_P24
   + 3 x_A_P25 + x_A_P26 + 3 x_A_P27 + x_A_P28 + 3 x_A_P29 + x_A_P30
   + 3 x_A_P31 + x_A_P32 + 3 x_A_P33 + x_A_P34 + 3 x_A_P35 + x_A_P36
   + 3 x_A_P37 + x_A_P38 + 3 x_A_P39 + x_A_P40 + 3 x_A_P41 + x_A_P42
   + 3 x_A_P43 + x_A_P44 + 3 x_A_P45 + x_A_P46 + 3 x_A_P47 + x_A_P48
   + 3 x_A_P49 + x_A_P50 + 3 x_A_P51 + x_A_P52 + 3 x_A_P53 + x_A_P54
   + 3 x_A_P55 + x_A_P56 + 3 x_A_P57 + x_A_P58 + 3 x_A_P59 + x_A_P60
   + 3 x_A_P61 + x_A_P62 <= 100
 cap_A_Machine_Hours: 3 x_A_P1 + 2 x_A_P2 + 2 x_A_P3 + 2 x_A_P4 + x_A_P5
   + 4 x_A_P6 + 3 x_A_P7 + 2 x_A_P8 + x_A_P9 + 4 x_A_P10 + 3 x_A_P11
   + 2 x_A_P12 + x_A_P13 + 4 x_A_P14 + 3 x_A_P15 + 2 x_A_P16 + x_A_P17
   + 4 x_A_P18 + 3 x_A_P19 + 2 x_A_P20 + x_A_P21 + 4 x_A_P22 + 3 x_A_P23
   + 2 x_A_P24 + x_A_P25 + 4 x_A_P26 + 3 x_A_P27 + 2 x_A_P28 + x_A_P29
   + 4 x_A_P30 + 3 x_A_P31 + 2 x_A_P32 + x_A_P33 + 4 x_A_P34 + 3 x_A_P35
   + 2 x_A_P36 + x_A_P37 + 4 x_A_P38 + 3 x_A_P39 + 2 x_A_P40 + x_A_P41
   + 4 x_A_P42 + 3 x_A_P43 + 2 x_A_P44 + x_A_P45 + 4 x_A_P46 + 3 x_A_P47
   + 2 x_A_P48 + x_A_P49 + 4 x_A_P50 + 3 x_A_P51 + 2 x_A_P52 + x_A_P53
   + 4 x_A_P54 + 3 x_A_P55 + 2 x_A_P56 + x_A_P57 + 4 x_A_P58 + 3 x_A_P59
   + 2 x_A_P60 + x_A_P61 + 4 x_A_P62 <= 90
 cap_A_Raw_Material_Units: 4 x_A_P1 + 3 x_A_P2 + 2 x_A_P3 + 3 x_A_P4
   + 4 x_A_P5 + x_A_P6 + 2 x_A_P7 + 3 x_A_P8 + 4 x_A_P9 + x_A_P10
   + 2 x_A_P11 + 3 x_A_P12 + 4 x_A_P13 + x_A_P14 + 2 x_A_P15 + 3 x_A_P16
   + 4 x_A_P17 + x_A_P18 + 2 x_A_P19 + 3 x_A_P20 + 4 x_A_P21 + x_A_P22
   + 2 x_A_P23 + 3 x_A_P24 + 4 x_A_P25 + x_A_P26 + 2 x_A_P27 + 3 x_A_P28
   + 4 x_A_P29 + x_A_P30 + 2 x_A_P31 + 3 x_A_P32 + 4 x_A_P33 + x_A_P34
   + 2 x_A_P35 + 3 x_A_P36 + 4 x_A_P37 + x_A_P38 + 2 x_A_P39 + 3 x_A_P40
   + 4 x_A_P41 + x_A_P42 + 2 x_A_P43 + 3 x_A_P44 + 4 x_A_P45 + x_A_P46
   + 2 x_A_P47 + 3 x_A_P48 + 4 x_A_P49 + x_A_P50 + 2 x_A_P51 + 3 x_A_P52
   + 4 x_A_P53 + x_A_P54 + 2 x_A_P55 + 3 x_A_P56 + 4 x_A_P57 + x_A_P58
   + 2 x_A_P59 + 3 x_A_P60 + 4 x_A_P61 + x_A_P62 <= 80
 cap_B_Labor_Hours: x_B_P1 + 2 x_B_P2 + 3 x_B_P3 + x_B_P4 + 2 x_B_P5
   + 3 x_B_P6 + 4 x_B_P7 + x_B_P8 + 2 x_B_P9 + 3 x_B_P10 + 4 x_B_P11
   + x_B_P12 + 2 x_B_P13 + 3 x_B_P14 + 4 x_B_P15 + x_B_P16 + 2 x_B_P17
   + 3 x_B_P18 + 4 x_B_P19 + x_B_P20 + 2 x_B_P21 + 3 x_B_P22 + 4 x_B_P23
   + x_B_P24 + 2 x_B_P25 + 3 x_B_P26 + 4 x_B_P27 + x_B_P28 + 2 x_B_P29
   + 3 x_B_P30 + 4 x_B_P31 + x_B_P32 + 2 x_B_P33 + 3 x_B_P34 + 4 x_B_P35
   + x_B_P36 + 2 x_B_P37 + 3 x_B_P38 + 4 x_B_P39 + x_B_P40 + 2 x_B_P41
   + 3 x_B_P42 + 4 x_B_P43 + x_B_P44 + 2 x_B_P45 + 3 x_B_P46 + 4 x_B_P47
   + x_B_P48 + 2 x_B_P49 + 3 x_B_P50 + 4 x_B_P51 + x_B_P52 + 2 x_B_P53
   + 3 x_B_P54 + 4 x_B_P55 + x_B_P56 + 2 x_B_P57 + 3 x_B_P58 + 4 x_B_P59
   + x_B_P60 + 2 x_B_P61 + 3 x_B_P62 <= 80
 cap_B_Machine_Hours: 4 x_B_P1 + 3 x_B_P2 + 2 x_B_P3 + 3 x_B_P4 + 2 x_B_P5
   + x_B_P6 + 4 x_B_P7 + 3 x_B_P8 + 2 x_B_P9 + x_B_P10 + 4 x_B_P11
   + 3 x_B_P12 + 2 x_B_P13 + x_B_P14 + 4 x_B_P15 + 3 x_B_P16 + 2 x_B_P17
   + x_B_P18 + 4 x_B_P19 + 3 x_B_P20 + 2 x_B_P21 + x_B_P22 + 4 x_B_P23
   + 3 x_B_P24 + 2 x_B_P25 + x_B_P26 + 4 x_B_P27 + 3 x_B_P28 + 2 x_B_P29
   + x_B_P30 + 4 x_B_P31 + 3 x_B_P32 + 2 x_B_P33 + x_B_P34 + 4 x_B_P35
   + 3 x_B_P36 + 2 x_B_P37 + x_B_P38 + 4 x_B_P39 + 3 x_B_P40 + 2 x_B_P41
   + x_B_P42 + 4 x_B_P43 + 3 x_B_P44 + 2 x_B_P45 + x_B_P46 + 4 x_B_P47
   + 3 x_B_P48 + 2 x_B_P49 + x_B_P50 + 4 x_B_P51 + 3 x_B_P52 + 2 x_B_P53
   + x_B_P54 + 4 x_B_P55 + 3 x_B_P56 + 2 x_B_P57 + x_B_P58 + 4 x_B_P59
   + 3 x_B_P60 + 2 x_B_P61 + x_B_P62 <= 100
 cap_B_Raw_Material_Units: 3 x_B_P1 + 2 x_B_P2 + x_B_P3 + 4 x_B_P4
   + 2 x_B_P5 + 4 x_B_P6 + 2 x_B_P7 + 4 x_B_P8 + 2 x_B_P9 + 4 x_B_P10
   + 2 x_B_P11 + 4 x_B_P12 + 2 x_B_P13 + 4 x_B_P14 + 2 x_B_P15 + 4 x_B_P16
   + 2 x_B_P17 + 4 x_B_P18 + 2 x_B_P19 + 4 x_B_P20 + 2 x_B_P21 + 4 x_B_P22
   + 2 x_B_P23 + 4 x_B_P24 + 2 x_B_P25 + 4 x_B_P26 + 2 x_B_P27 + 4 x_B_P28
   + 2 x_B_P29 + 4 x_B_P30 + 2 x_B_P31 + 4 x_B_P32 + 2 x_B_P33 + 4 x_B_P34
   + 2 x_B_P35 + 4 x_B_P36 + 2 x_B_P37 + 4 x_B_P38 + 2 x_B_P39 + 4 x_B_P40
   + 2 x_B_P41 + 4 x_B_P42 + 2 x_B_P43 + 4 x_B_P44 + 2 x_B_P45 + 4 x_B_P46
   + 2 x_B_P47 + 4 x_B_P48 + 2 x_B_P49 + 4 x_B_P50 + 2 x_B_P51 + 4 x_B_P52
   + 2 x_B_P53 + 4 x_B_P54 + 2 x_B_P55 + 4 x_B_P56 + 2 x_B_P57 + 4 x_B_P58
   + 2 x_B_P59 + 4 x_B_P60 + 2 x_B_P61 + 4 x_B_P62 <= 90
Bounds
End
