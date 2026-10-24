{
  "query": "A digital game store needs to decide which games to list on different platforms, considering that these games belong to various genres such as racing, sports, and others. Each platform has a limited memory capacity, with specific details provided in \"capacity.csv.\" The predefined value and memory requirement of each game are available in \"products.csv.\" The objective is to determine which genres and how many units of each game to list on each platform to maximize the total value of the games across all platforms, while ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  x_ij represent the number of units of game j to be listed on platform i.The decision variables must be integers.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "PlatformID",
        "Capacity"
      ],
      "file_name": "capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "995",
            "PlatformID": "1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Capacity": "1143",
            "PlatformID": "2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Capacity": "949",
            "PlatformID": "3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Capacity": "969",
            "PlatformID": "4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Capacity": "1649",
            "PlatformID": "5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Capacity": "870",
            "PlatformID": "6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Capacity": "1064",
            "PlatformID": "7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Capacity": "536",
            "PlatformID": "8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Capacity": "766",
            "PlatformID": "9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Capacity": "532",
            "PlatformID": "10"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Capacity": "1703",
            "PlatformID": "11"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Capacity": "1633",
            "PlatformID": "12"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Capacity": "1203",
            "PlatformID": "13"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Capacity": "1979",
            "PlatformID": "14"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "Capacity": "1797",
            "PlatformID": "15"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "ProductName",
        "Value",
        "Weight"
      ],
      "file_name": "products.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ProductName": "Racing",
            "Value": "59",
            "Weight": "776"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ProductName": "Sports",
            "Value": "83",
            "Weight": "573"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ProductName": "Action",
            "Value": "94",
            "Weight": "127"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ProductName": "Adventure",
            "Value": "41",
            "Weight": "138"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ProductName": "RPG",
            "Value": "96",
            "Weight": "385"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ProductName": "Shooter",
            "Value": "12",
            "Weight": "263"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ProductName": "Strategy",
            "Value": "83",
            "Weight": "473"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ProductName": "Simulation",
            "Value": "36",
            "Weight": "387"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ProductName": "Puzzle",
            "Value": "56",
            "Weight": "390"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ProductName": "Fighting",
            "Value": "27",
            "Weight": "556"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ProductName": "Platformer",
            "Value": "47",
            "Weight": "601"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ProductName": "Survival",
            "Value": "24",
            "Weight": "441"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ProductName": "Horror",
            "Value": "14",
            "Weight": "603"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ProductName": "Sandbox",
            "Value": "22",
            "Weight": "411"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ProductName": "MMO",
            "Value": "17",
            "Weight": "652"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
