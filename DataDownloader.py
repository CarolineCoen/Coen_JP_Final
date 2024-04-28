from City import City
import numpy as np
import urllib.parse
import urllib.request
import ssl
from playwright.sync_api import sync_playwright
import time

# convert the file of the top 249 cities in West Africa into an array of City objects
def makeCities():
    citiesText = open('cities.txt', 'r')
    fileContent = citiesText.read()
    fileContent = fileContent.split()
    numWords = len(fileContent)
    cities = np.empty(249, dtype=object)
    r = 0
    j = 0
    for i in range(249):
        cityName = fileContent[r]
        r = r+1
        cityCountry = fileContent[r]
        r = r+1
        cityLatitude = fileContent[r]
        r = r+1
        cityLongitude = fileContent[r]
        r = r+1
        city = City(name=cityName, country=cityCountry, latitude=cityLatitude, longitude=cityLongitude)
        cities[j] = city
        j=j+1
    return cities

# save the list of the urls for the screenshots we want to take, and also the coordinates of those screenshots
# so that we can acquire training data from the areas around those coordinates
def createURLs(cities):

    urls = open('urls.txt', 'w')
    coordinates = open('coordinates.txt', 'w')

    # take screenshots for each city in cities
    for c in range(len(cities)):

        # Take a screenshot of the Google Maps frame approximately 30 kilometers north, south, east and west of
        # the center of each city
        baseLatitude = float(cities[c].latitude)
        baseLongitude = float(cities[c].longitude)
        for i in range (4):
            latitude = baseLatitude
            longitude = baseLongitude
            # 30 kilometers is approximately 0.3 degrees latitude and longitude
            # The number of kilometers 0.3 degrees represents is dependent on curvature
            # of the earth at that location
            if (i == 0):
                latitude = latitude + 0.3
            elif (i == 1):
                latitude = latitude - 0.3
            elif (i == 2):
                longitude = longitude + 0.3
            else:
                longitude = longitude - 0.3
		
		    # create the url for this latitude and longitude:
            url = 'https://www.google.com/maps/@'
            url = url + '{}'.format(latitude)
            url = url + ','
            url = url + '{}'.format(longitude)
            url = url + ','
            url = url + '1648m/data=!3m1!1e3!5m1!1e4?entry=ttu'

            urls.write(url)
            urls.write("\n")
            coordinates.write(cities[c].name)
            coordinates.write(' ')
            coordinates.write(str(latitude))
            coordinates.write(' ')
            coordinates.write(str(longitude))
            coordinates.write("\n")

    urls.close()
    coordinates.close()

# Citation: [4]
def takeScreenshots(playwright, cities):
    # launch the browser
    browser = playwright.chromium.launch()

    readUrls = open('urls.txt', 'r')

    z = 1
    y = -1
    for x in range(996):
        # pulling each
        thisUrl = readUrls.readline()
        # opens a new browser page
        website = browser.new_page()
        # navigate to the website
        website.goto(thisUrl)

        # pause so that the Google Maps page has time to fully load
        time.sleep(10)

        if ((x % 4) == 0):
            y=y+1
            z = 1
        name = cities[y].name

        # take a full-page screenshot
        website.screenshot(path='./screenshots/%s/%s%d.png'%(name, name, z), full_page=True)

        # the images that end up looking weird are over the ocean, and you can't zoom into such a level over the ocean
        z = z+1
        
    # always close the browser
    browser.close()

def main():

    cities = makeCities()
    createURLs(cities)
    with sync_playwright() as playwright:
        takeScreenshots(playwright, cities)

main()