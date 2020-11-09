from data_weighted_kmeans import *
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import csv
from haversine import haversine
import random

def find_nearest_zip(points,centers):
    for c in centers:
        clat=c['coords'][1]
        clong=c['coords'][0]
        ds=[]
        for p in points:
            plat=p['coords'][1]
            plong=p['coords'][0]
            d=distance(clat,clong,plat,plong)
            ds.append(d)
        idx = np.argmin(np.array(ds))
    return centers

random.seed(42)
k=2

points=[]
with open("us_census2.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(reader)
    for row in reader:
        d = dict(zip(header,row))
        points.append({"coords": np.array([float(d['longitude']),float(d['latitude'])]),"w":int(d['population'])})

points = random.sample(points,7)
centers = equally_spaced_initial_clusters(points,k)

points,centers,it_num = data_weighted_kmeans(points,centers,k)

centers=find_nearest_zip(points,centers)

print ("k,latitude,longitude,population,c,clat,clong,cdistance")

#create dictionary of centers keyed off their ID
d={}
for i,c in enumerate(centers):
    d[i]=c

file = open("csvresults.csv", 'w')
pts = []
ctrs = []

for p in points:
    p1=p["coords"]
    p2=d[p['c']]['coords']
    dist=str(int(distance(p1[1],p1[0],p2[1],p2[0])))
    out=[k]
    tmpPoint = [p["coords"][1], p["coords"][0], p['w']]
    pts.append(tmpPoint)
    out.append(p["coords"][1]) #lat
    out.append(p["coords"][0]) #long
    out.append(p['w'])
    tmpCenter = [d[p['c']]['coords'][1], d[p['c']]['coords'][0]]
    ctrs.append(tmpCenter)
    #out.append(p['c'])
    #out.append(d[p['c']]['coords'][1]) #lat
    #out.append(d[p['c']]['coords'][0]) #long
    #out.append(dist)
    print (",".join([str(s) for s in out]), file=file)

file.close()

print(pts)
print(ctrs)

img = plt.imread("worldmap800.jpg")
fig, ax = plt.subplots()
#ax.imshow(img)
for pt in pts:
    ax.scatter(pt[0], pt[1], s=pt[2]/1000, c="blue")

for ctr in ctrs:
    ax.scatter(ctr[0], ctr[1], s=100, c="red")

#ax.scatter([800*(18.180103+180)/(2*180),800*(18.180103+180)/(2*180), 340*(-66.74947+180)/(2*180)], [2, 1, 1.5])
plt.show()
