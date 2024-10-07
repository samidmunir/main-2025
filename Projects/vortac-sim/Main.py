import pygame as pg
import time
import Aircraft as AC

# CONSTANTS
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 800
FPS = 60 # Frame rate.

class Main:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
        pg.display.set_caption('VORTAC SIM - Air Traffic Control Radar')
        
        # Create an aircraft object.
        self.aircraft = AC.Aircraft('SDM147', 'B77W', 100.0, 100.0, 250.0, 15000.0, 90.0)
        
        # Add a sample plan for the aircraft.
        self.aircraft.add_to_plan(400.0, 400.0, 17000.0, 300.0, 180.0, 5000)
        
        # Pygame clock for managing FPS.
        self.clock = pg.time.Clock()
        
    def run(self):
        running = True
        last_time = time.time()
        
        while running:
            dt = (time.time() - last_time) * 1000 # time difference in ms.
            last_time = time.time()
            
            # Handle events.
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            # Update the aircraft's state.
            self.aircraft.update(dt)
            
            # Draw the aircraft on the canvas.
            self.screen.fill('#262626') # Clear the canvas.
            if self.aircraft.is_active:
                self.draw_aircraft(self.aircraft)
            
            # Update the display.
            # pg.display.flip()
            
            # Update the canvas and wait for the next frame.
            pg.display.update()
            self.clock.tick(FPS)
        
        pg.quit()
    
    def draw_aircraft(self, aircraft):
        x = int(aircraft.x_loc)
        y = int(aircraft.y_loc)
        size = 10
        pg.draw.polygon(self.screen, (255, 255, 255), [(x, y - size), (x - size, y + size), (x + size, y + size)])
        
        font = pg.font.SysFont(None, 24)
        flight_number_surface = font.render(aircraft.flight_number, True, (255, 255, 255))
        self.screen.blit(flight_number_surface, (x - 15, y - 15))
        
if __name__ == '__main__':
    sim = Main()
    sim.run()