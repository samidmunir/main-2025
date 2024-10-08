from typing import List


class Airport:
    def __init__(self, name: str, city: str, country: str, loc_x: float, loc_y: float, number_of_runways: int, runways: List, active: bool = True):
        print('\nAirport.__init__() called -->')
        if name is None or len(name) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid name. Name must be at least 4 characters long.')
            return
        self.name = name
        if city is None or len(city) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid city. City name must be at least 3 characters long.')
            return
        self.city = city
        if country is None or len(country) < 4:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid country. Country name must be at least 3 characters long.')
            return
        self.country = country
        self.loc_x = loc_x
        self.loc_y = loc_y
        if number_of_runways <= 0:
            print('***ERROR***: Airport initialization failed')
            print('\tInvalid number of runways. Number of runways must be greater than 0.')
            return
        self.number_of_runways = number_of_runways
        self.runways = runways
        self.active = active
        print('\tAirport initialized successfully.')
    
    def update_name(self, new_name: str):
        print('\nAirport.update_name() called -->')
        if new_name is None or len(new_name) < 4:
            print('***ERROR***: Failed to update airport name')
            print('\tInvalid name. Name must be at least 4 characters long.')
            return
        self.name = new_name
        print('\tAirport name updated successfully.')
    
    def update_location(self, new_city: str, new_country: str):
        print('\nAirport.update_location() called -->')
        if new_city is None or len(new_city) < 4:
            print('***ERROR***: Failed to update airport city')
            print('\tInvalid city. City name must be at least 3 characters long.')
            return
        if new_country is None or len(new_country) < 4:
            print('***ERROR***: Failed to update airport country')
            print('\tInvalid country. Country name must be at least 3 characters long.')
            return
        self.city = new_city
        self.country = new_country
        print('\tAirport location updated successfully.')
    
    def update_coordinates(self, new_loc_x: float, new_loc_y: float):
        print('\nAirport.update_coordinates() called -->')
        if new_loc_x is None or new_loc_y is None:
            print('***ERROR***: Failed to update airport coordinates')
            print('\tInvalid coordinates. Coordinates must be numeric values.')
            return
        self.loc_x = new_loc_x
        self.loc_y = new_loc_y
        print('\tAirport coordinates updated successfully.')
    
    def update_number_of_runways(self, new_number_of_runways: int):
        print('\nAirport.update_number_of_runways() called -->')
        if new_number_of_runways <= 0:
            print('***ERROR***: Failed to update airport number of runways')
            print('\tInvalid number of runways. Number of runways must be greater than 0.')
            return
        self.number_of_runways = new_number_of_runways
        print('\tAirport number of runways updated successfully.')