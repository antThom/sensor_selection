from osm import OSMDownloader

osm = OSMDownloader(dist=1000)

buildings = osm.buildings_from_place("Atlanta, Georgia, USA")

nodes, roads = osm.roads_from_place("Atlanta, Georgia, USA")

water = osm.water_from_place("Atlanta, Georgia, USA")

parks = osm.parks_from_place("Atlanta, Georgia, USA")
