class Runway:
    def __init__(self, name: str, length: float, heading: float, start_x_loc: float, start_y_loc: float, end_x_loc: float, end_y_loc: float, active: bool = True):
        print('\nRunway.__init__() called -->')
        if name is None or len(name) < 2:
            print('***ERROR***: Runway initialization failed')
            print('\tInvalid name. Name must be at least 2 characters long.')
            return
        self.name = name
        self.length = length
        self.heading = heading
        self.start_x_loc = start_x_loc
        self.start_y_loc = start_y_loc
        self.end_x_loc = end_x_loc
        self.end_y_loc = end_y_loc
        self.active = active
        print('\tRunway initialized successfully.')
    
    def update_name(self, new_name: str):
        print('\nRunway.update_name() called -->')
        if new_name is None or len(new_name) < 2:
            print('***ERROR***: Failed to update runway name')
            print('\tInvalid name. Name must be at least 2 characters long.')
            return
        self.name = new_name
        print('\tRunway name updated successfully.')
    
    def update_length(self, new_length: float):
        print('\nRunway.update_lenght() called -->')
        if new_length is None or new_length <= 0:
            print('***ERROR***: Failed to update runway length')
            print('\tInvalid length. Length must be a positive numeric value.')
            return
        self.lenght = new_length
        print('\tRunway length updated successfully.')
    
    def update_heading(self, new_heading: float):
        print('\nRunway.update_heading() called -->')
        if new_heading is None or (new_heading < 0 or new_heading > 360):
            print('***ERROR***: Failed to update runway heading')
            print('\tInvalid heading. Heading must be between 0 and 360 degrees.')
            return
        self.heading = new_heading
        print('\tRunway heading updated successfully.')