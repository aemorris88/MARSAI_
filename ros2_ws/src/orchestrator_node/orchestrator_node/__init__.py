self.dispatcher = DroneDispatcher(self, drone_ids=["drone_1","drone_2","drone_3"])
self.dispatcher.assign("scout", search_area={"x_min":0,"x_max":20,"y_min":-18,"y_max":15})