class City:
    def __init__(self, name, country, latitude, longitude):
        self.name = name
        self.country = country
        self.latitude = latitude
        self.longitude = longitude

    def getLatitude(self):
        return self.latitude
    
    def getLongitude(self):
        return self.longitude
    
    def getName(self):
        return self.name
    
    def __str__(self) -> str:
        return "{}, {}: {},{}".format(self.name, self.country, self.latitude, self.longitude)
