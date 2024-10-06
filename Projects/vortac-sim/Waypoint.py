import time
class Waypoint:
    def __init__(self, name: str, x_loc: float, y_loc: float, active: bool = True):
        print('\nWaypoint.__init__() called -->')
        if name is None or len(name) < 3:
            time.sleep(2)
            print('***ERROR***: Waypoint initialization failed')
            print('\tInvalid name. Name must be at least 3 characters long.')
        self.name = name
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.active = active
        print('\tWaypoint initialized successfully.')
        
    
    def update_name(self, new_name: str):
        print('\nWaypoint.update_name() called -->')
        if new_name is None or len(new_name) < 3:
            time.sleep(2)
            print('***ERROR***: Failed to update waypoint name')
            print('\tInvalid name. Name must be at least 3 characters long.')
        self.name = new_name
        print('\tWaypoint name updated successfully.')
    
    def update_x_loc(self, new_x_loc: float):
        print('\nWaypoint.update_x_loc() called -->')
        if new_x_loc is None:
            time.sleep(2)
            print('***ERROR***: Failed to update waypoint x-location')
            print('\tInvalid x-location. Location must be a numeric value.')
        self.x_loc = new_x_loc
        print('\tWaypoint x-location updated successfully.')
    
    def update_y_loc(self, new_y_loc: float):
        print('\nWaypoint.update_y_loc() called -->')
        if new_y_loc is None:
            time.sleep(2)
            print('***ERROR***: Failed to update waypoint y-location')
            print('\tInvalid y-location. Location must be a numeric value.')
        self.y_loc = new_y_loc
        print('\tWaypoint y-location updated successfully.')
    
    def update_is_active(self, new_active: bool):
        print('\nWaypoint.update_is_active() called -->')
        self.active = new_active
        print('\tWaypoint active status updated successfully.')
        
    def display_waypoint_info(self):
        print('\nWaypoint.display_waypoint_info() called -->')
        print(f'\tName: {self.name}')
        print(f'\tLocation: ({self.x_loc}, {self.y_loc})')
        print(f'\tActive: {self.active}')