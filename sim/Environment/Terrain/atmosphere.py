from sim.Environment.Terrain.sky import SKY


class ATMOSPHERE:
    def __init__(self, loader, config, render):
        for key in config:
            if key == "sky":
                self.sky = SKY(
                    config=config[key],
                    loader=loader,
                    render=render,
                    day_length=self.day_length,
                    time_of_day=self.time_of_day,
                )
            elif key == "time":
                self.day_length = config[key].get("day_length", 24)
                self.time_of_day = config[key].get("time", "08:00:00")
            # elif key == "light_source":
            #     self.light_source = SUN(config=config[key], loader=loader, render=render, day_length=self.day_length, time_of_day=self.time_of_day)
            #     # self.sun = SUN(config=config[key], loader=loader, render=render, day_length=self.day_length)
