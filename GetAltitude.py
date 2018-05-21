# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:01:13 2018

@author: Amanda
"""

#Code to automatically extract MSL elevation for a given lat/lon
#Original code here: https://mavicpilots.com/threads/altitude-information-from-exif-data-photos.32535/

import urllib
import json
import exifread
import sys

# get degrees from GPS EXIF tag
def degress(tag):
    d = float(tag.values[0].num) / float(tag.values[0].den)
    m = float(tag.values[1].num) / float(tag.values[1].den)
    s = float(tag.values[2].num) / float(tag.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

# must provide a file name
if (len(sys.argv)!=2):
    raise Exception("No file name provided")
filename = sys.argv[1]

# read the exif tags
with open(filename, 'rb') as f:
    tags = exifread.process_file(f)

# get lat/lon
lat = degress(tags["GPS GPSLatitude"])
lon = degress(tags["GPS GPSLongitude"])
lat = -lat if tags["GPS GPSLatitudeRef"].values[0]!='N' else lat
lon = -lon if tags["GPS GPSLatitudeRef"].values[0]!='E' else lon

# get ground elevation at this location, if possible
try:
    ground_level = float("nan")
    open_elevation_reply = json.loads(urllib.request.urlopen("https://api.open-elevation.com/api/v1/lookup?locations=%f,%f" % (lat,lon)).read())
    ground_level = float(open_elevation_reply["results"][0]["elevation"])
except:
    pass

# get the altitude
alt = tags["GPS GPSAltitude"]
alt = float(alt.values[0].num) / float(alt.values[0].den)
below_sea_level = tags["GPS GPSAltitudeRef"].values[0]!=0;
alt = -alt if below_sea_level else alt
agl = alt - ground_level

# spit it out
print("Latitude[deg]     : %f" % lat)
print("Longitude[deg]    : %f" % lon)
print("Altitude [m, ASL] : %f" % alt)
print("Altitude [m, AGL] : %f" % agl)