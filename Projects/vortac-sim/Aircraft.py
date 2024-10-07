import math
from Main import CANVAS_WIDTH, CANVAS_HEIGHT

class Aircraft:
    def __init__(self, flight_number: str, type: str, x_loc: float, y_loc: float, speed: float, altitude: float, heading: float):
        print(f'\nAircraft.__init__() called -->')
        if flight_number is None or len(flight_number) < 3:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid flight number. Flight number must be at least 3 characters long.')
            return
        self.flight_number = flight_number
        if type is None or len(type) < 3:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid aircraft type. Type must be at least 3 characters long.')
            return
        self.type = type
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.speed = speed
        self.altitude = altitude
        if heading < 0 or heading > 360:
            print('***ERROR***: Aircraft initialization failed')
            print('\tInvalid heading. Heading must be between 0 and 360 degrees.')
            return
        self.heading = heading
        self.plan = []
        self.elapsed_time = 0.0
        self.is_active = True
        print(f'\tAircraft initialized successfully [{self.flight_number}]')
    
    def lerp(start, end, t):
            return start + t * (end - start)
    
    def update(self, dt):
        if not self.is_active:
            return # No need to update if the aircraft is inactive.
        
        if self.plan:
            # Handle movement based on plan.
            target_x, target_y, target_altitude, target_speed, target_heading, duration = self.plan[0]
            t = min(1, self.elapsed_time / duration)
            
            self.x_loc = lerp(self.x_loc, target_x, t)
            self.y_loc = lerp(self.y_loc, target_y, t)
            self.altitude = lerp(self.altitude, target_altitude, t)
            self.speed = lerp(self.speed, target_speed, t)
            
            delta_heading = (target_heading - self.heading) % 360
            if delta_heading > 180:
                delta_heading -= 360
            self.heading = lerp(self.heading, self.heading + delta_heading , t) % 360
            
            self.elapsed_time += dt
            if self.elapsed_time >= duration:
                self.plan.pop(0)
                self.elapsed_time = 0.0
        else:
            # Continue moving in the last known state (constant velocity motion)
            dx = self.speed * math.cos(math.radians(self.heading)) * dt / 1000
            dy = self.speed * math.sin(math.radians(self.heading)) * dt / 1000
            self.x_loc += dx
            self.y_loc += dy
        
        # Check if the aircraft is out of bounds (off the canvas).
        if not (0 <= self.x_loc <= CANVAS_WIDTH) and (0 <= self.y_loc <= CANVAS_HEIGHT):
            self.is_active = False # Mark aircraft as inactive from VORTAC canvas.
    
    def add_to_plan(self, target_x, target_y, target_altitude, target_speed, target_heading, duration):
        self.plan.append((target_x, target_y, target_altitude, target_speed, target_heading, duration))