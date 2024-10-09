class FlightPlan:
    def __init__(self, flight_number: str, departure_airport: str, destination: str, plan: list):
        self.flight_number = flight_number
        self.departure_airport = departure_airport
        self.destination = destination
        self.plan = plan
        self.plan_length = len(plan)
        self.current_index = 0
        print('\nFlightPlan.__init__() called -->')