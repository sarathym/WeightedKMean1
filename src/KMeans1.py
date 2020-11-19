from data_weighted_kmeans import *
from mpl_toolkits.basemap import Basemap
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import csv
from haversine import haversine
import random

max = 0
min = 1

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
k=200

points=[]
with open("us_census3.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(reader)
    for row in reader:
        d = dict(zip(header,row))
        points.append({"coords": np.array([float(d['longitude']),float(d['latitude'])]),"w":int(d['population'])})

points = random.sample(points, 5000)
centers = equally_spaced_initial_clusters(points,k)

points,centers,it_num = data_weighted_kmeans(points,centers,k)

centers=find_nearest_zip(points,centers)

#create dictionary of centers keyed off their ID
d={}
for i,c in enumerate(centers):
    d[i]=c

file = open("csvresults.csv", 'w')
pts = []
ctrs = []
ctrWeights = {}
ctrDeviation = {}

for p in points:
    p1=p["coords"]
    p2=d[p['c']]['coords']
    dist=str(int(distance(p1[1],p1[0],p2[1],p2[0])))
    out=[k]
    tmpPoint = (p["coords"][1], p["coords"][0], p['w'])
    pts.append(tmpPoint)
    out.append(p["coords"][1]) #lat
    out.append(p["coords"][0]) #long
    out.append(p['w'])
    tmpCenter = (d[p['c']]['coords'][1], d[p['c']]['coords'][0])
    ctrs.append(tmpCenter)
    if tmpCenter in ctrWeights:
        tmp = ctrWeights[tmpCenter]
        # tmp += p['w'] add weight
        tmp += 1
        ctrWeights[tmpCenter] = tmp
        if tmp > max:
            max = tmp
        if tmp < min:
            min = tmp
    else:
        ctrWeights[tmpCenter] = 1
    deviation = math.sqrt(math.pow((p["coords"][1] - tmpCenter[0]), 2) + math.pow((p["coords"][0] - tmpCenter[1]), 2))
    if tmpCenter in ctrDeviation:
        tmpDeviation = ctrDeviation[tmpCenter]
        tmp1 = tmpDeviation[0]
        tmp2 = tmpDeviation[1]
        tmp1 += deviation
        tmp2 += 1
        ctrDeviation[tmpCenter] = (tmp1, tmp2)
    else:
        ctrDeviation[tmpCenter] = (deviation, 1)
    print (",".join([str(s) for s in out]), file=file)

file.close()

#img = plt.imread("worldmap800.jpg")
#fig, ax = plt.subplots()
#ax.imshow(img)

#mapWidth = 800
#mapHeight = 340

#for pt in pts:
#    ax.scatter(pt[0], pt[1], s=pt[2]/1000, c="blue", edgecolors="blue")

#for ctr in ctrs:
#    ax.scatter(ctr[0], ctr[1], s=ctrWeights.get(ctr)/5000, c="red", edgecolors="yellow")

#plt.show()

fig = plt.figure(figsize=(8,4.5))
m = Basemap(projection='robin',lon_0=0,resolution='h')
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='gray',lake_color='white')

index=0
for pt in pts:
    lat = pt[0]
    long = pt[1]
    x,y = m(long, lat)
    plt.scatter(x, y, s=pt[2]/5000, c="blue", edgecolors="blue", zorder = 3)

for ctr in ctrs:
    lat = ctr[0]
    long = ctr[1]
    x,y = m(long, lat)
    # plt.scatter(x, y, s=ctrWeights.get(ctr)/25000, c="red", edgecolors="yellow", zorder = 3)
    normalized = (ctrWeights.get(ctr) - min) / (max - min)
    plt.scatter(x, y, c="red", s = (ctrDeviation[ctr][0]/ctrDeviation[ctr][1])*10, zorder = 3, alpha = normalized)

plt.show()
