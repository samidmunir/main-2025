from typing import List


class Airport:
    def __init__(self, name: str, city: str, country: str, loc_x: float, loc_y: float, number_of_runways: int, runways: List, active: bool = True):
        print('\nAirport.__init__() called -->')
        if name is None or len(name) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid name. Name must be at least 4 characters long.')
        self.name = name
        if city is None or len(city) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid city. City name must be at least 3 characters long.')
        self.city = city
        if country is None or len(country) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid country. Country name must be at least 3 characters long.')
        self.country = country
        self.loc_x = loc_x
        self.loc_y = loc_y
        if number_of_runways <= 0:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid number of runways. Number of runways must be greater than 0.')
        self.number_of_runways = number_of_runways
        self.runways = runways
        self.active = active
        print('\tAirport initialized successfully.')