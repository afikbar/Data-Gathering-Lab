#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.elasticsearch:elasticsearch-hadoop:7.4.2 pyspark-shell'

# IP = 'da2019w-1019.eastus.cloudapp.azure.com'
IP = '10.0.0.25'
# GOOGLE_API_KEY = 'AIzaSyBSp6bqrg9ijhLKXAkn5Rt4BrPpnnpv2d8'
HERE_API_KEY = 'TarqgkWPPRHbWkLVZWCz2VAGJVHWj_B18ii-yO5pyZo'

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from elasticsearch import Elasticsearch, helpers
from pyspark.sql.window import Window
import requests


spark = SparkSession     .builder     .appName("Python Spark SQL basic example")     .config("spark.executor.memory", "2g")     .config("spark.driver.memory", "1g")     .getOrCreate()

spark.conf.set("spark.sql.session.timeZone", "GMT")

es = Elasticsearch([{'host': IP}])


# ## Helper Functions:

# In[2]:



def read_elastic(index, query="", scroll_size="10000", array_field=""):
    if not es.indices.exists(index):
        raise Exception("Index doesn't exist!")

    return spark.read                .format("org.elasticsearch.spark.sql")                .option("es.nodes.wan.only","true")                .option("es.port","9200")                .option("es.nodes",IP)                .option("es.nodes.client.only", "false")                .option("pushdown", "true")                .option("es.query", query)                .option("es.scroll.size", scroll_size)                .option("es.scroll.keepalive", "120m")                .option("es.read.field.as.array.include", array_field)                .load(index)

        
DEFUALT_SCEHMA = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "actualDelay" : { "type": "long" },
            "areaId" : { "type": "long" },
            "areaId1" : { "type": "long" },
            "areaId2" : { "type": "long" },
            "areaId3" : { "type": "long" },
            "atStop" : { "type": "boolean" },
            "busStop" : { "type": "long" },
            "congestion" : { "type": "boolean" },
            "gridID" : { "type": "keyword" },
            "journeyPatternId" : { "type": "keyword" },
            "lineId" : { "type": "keyword" },
            "coordinates" : { "type": "geo_point" },
            "timestamp" : { "type": "date", "format" : "epoch_millis" },
            "vehicleId" : { "type": "long" },
            "dateTime" : { "type": "date" }
        }
    }
}

def write_to_elastic(df, index: str, settings=DEFUALT_SCEHMA, append=True):
    if es.indices.exists(index) and not append:
        es.indices.delete(index=index)
    
    es.indices.create(index=index, ignore=400, body=settings)

    df.write.format("org.elasticsearch.spark.sql")        .option("es.resource", index)        .option("es.nodes.wan.only","true")        .option("es.port","9200")        .option("es.nodes",IP)        .option("es.nodes.client.only", "false")        .save()



def calculate_centroids(df):
    centroid_df = df.groupBy('busStop')                    .agg(F.mean(df.coordinates[0]).alias('centroid_longitude'), 
                            F.mean(df.coordinates[1]).alias('centroid_latitude'))

    centroid_df = centroid_df.withColumn("coordinates", F.array('centroid_longitude', 'centroid_latitude'))                                .drop('centroid_longitude', 'centroid_latitude')
    return centroid_df

from math import radians, cos, sin, asin, sqrt

@F.udf("float")
def get_distance(coord_a, coord_b):
    longit_a, latit_a = coord_a
    longit_b, latit_b = coord_b
    if None in [longit_a, latit_a, longit_b, latit_b]:
        return 9999
    # Transform to radians
    longit_a, latit_a, longit_b, latit_b = map(radians, [longit_a,  latit_a, longit_b, latit_b])
    dist_longit = longit_b - longit_a
    dist_latit = latit_b - latit_a
    # Calculate area
    area = sin(dist_latit/2)**2 + cos(latit_a) * cos(latit_b) * sin(dist_longit/2)**2
    # Calculate the central angle
    central_angle = 2 * asin(sqrt(area))
    radius = 6371
    # Calculate Distance
    distance = central_angle * radius
    return abs(round(distance, 4))

def add_distance_to_centroid(centroid_df, stop_df, drop_centroid_col=True):
    c_df = centroid_df.selectExpr("coordinates as c_coordinates", "busStop as c_busStop")
    left_join = stop_df.join(c_df, stop_df['busStop'] == c_df['c_busStop'], how='inner')
    res = left_join.withColumn('distance', get_distance(left_join.c_coordinates, left_join.coordinates)).drop('c_busStop')
    if drop_centroid_col:
        return res.drop('c_coordinates')
    return res

@F.udf(ArrayType(DoubleType()))
def merge_coordinates(longitude, latitude):
    return [float(longitude), float(latitude)]

@F.udf("float")
def normalize_text_distance(name1, name2, distance):
    return distance / max(len(name1), len(name2))


# ## Save BusStop Information:

# In[ ]:


busStops = requests.get(f'https://data.smartdublin.ie/cgi-bin/rtpi/busstopinformation').json()['results']
columns_to_keep = ['stopid', 'displaystopid', 'shortname', 'shortnamelocalized',
                   'fullname', 'fullnamelocalized', 'longitude', 'latitude']
stops_df = spark.createDataFrame(pd.DataFrame(busStops).loc[:,columns_to_keep])
stops_df = stops_df.withColumn('coordinates', merge_coordinates(stops_df.longitude, stops_df.latitude)).drop('longitude', 'latitude')


# In[ ]:


stops_index = 'stop-information-index'
settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "stopid" : { "type": "long" },
            "displaystopid" : { "type": "string" },
            "shortname" : { "type": "string" },
            "shortnamelocalized" : { "type": "string" },
            "fullname" : { "type": "string" },
            "fullnamelocalized" : { "type": "string" },
            "coordinates" : {"type" : 'geo_point'}
        }
    }
}
write_to_elastic(stops_df, stops_index, settings, append=False)


# # Where am I?

# In[3]:


def lev_distance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

@F.udf('float')
def get_text_distance(station_name, reverse_gecode):
    lev_dist = 2**10
    
    if reverse_gecode and station_name:
        lev_dist = min([lev_distance(station_name, address)/max(len(station_name), len(address)) for address in reverse_gecode if address])
    return lev_dist


STOPWORDS = ['avenue', 'ave', 'blvd', 'boulevard', 'box', 'cir', 'court', 'ct', 'drive', 'dr', 'lane', 'ln', 'loop', 'lp', 'pl', 'place', 'po', 'pob', 'pt', 'rd', 'road', 'route', 'rr', 'rte', 'rural', 'sq', 'st', 'ste', 'street', 'suit', 'trl', 'way', 'wy']

def extract_address(result):
    address = []
    try:
        address = result['Location']['Address']['Street']
    except:
        address = result['Location']['Address']['Label']
    return ' '.join(filter(lambda word: word.lower().rstrip('.') not in STOPWORDS, address.split()))

@F.udf(ArrayType(StringType()))
def reverse_gecode(coords):
    lng, lat = coords
    params = {'prox' : f"{lat}, {lng}, 5", 'mode' : 'retrieveAddresses', 'apiKey' : HERE_API_KEY}
    results = requests.get("https://reverse.geocoder.ls.hereapi.com/6.2/reversegeocode.json", params=params)                                                                                    .json()['Response']['View'][0]['Result']
    addresses = list(set(map(extract_address, results)))
    
    return addresses


@F.udf(BooleanType())
def is_approx_near(coords_a, coords_b, decimal=5):
    lng_a, lat_a = coords_a
    lng_b, lat_b = coords_b

    return (round(lng_a, decimal) == round(lng_b, decimal)) and (round(lat_a, decimal) == round(lat_b, decimal))


# In[5]:


agg_stop_df = read_elastic('agg1-coords-index')                .withColumn('reverse_gecode', reverse_gecode(F.col('coordinates')))
                # .withColumnRenamed('coordinates', 'agg_coords')
settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {    
            "coordinates" : { "type": "geo_point" },
            "reverse_gecode" : { "type": "keyword" }
        }
    }
}
# Was "reverse-gecode-index"
write_to_elastic(agg_stop_df, 'agg1-street-index', settings= settings, append= False)


# In[8]:


agg_stop_df =  read_elastic("agg1-street-index", array_field="reverse_gecode")                .withColumnRenamed('coordinates', 'agg_coords')
                # .withColumn('reverse_gecode', F.array_distinct("reverse_gecode"))

stop_df = read_elastic('stop-index')




reverse_gecode_df = stop_df.join(agg_stop_df, 
                                    (F.round(F.element_at(stop_df.coordinates, 1), 5) == F.round(F.element_at(agg_stop_df.agg_coords, 1), 5)) &
                                    (F.round(F.element_at(stop_df.coordinates, 2), 5) == F.round(F.element_at(agg_stop_df.agg_coords, 2), 5)),
                                    how='left').drop('agg_coords')

settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "actualDelay" : { "type": "long" },
            "areaId" : { "type": "long" },
            "areaId1" : { "type": "long" },
            "areaId2" : { "type": "long" },
            "areaId3" : { "type": "long" },
            "atStop" : { "type": "boolean" },
            "busStop" : { "type": "long" },
            "congestion" : { "type": "boolean" },
            "gridID" : { "type": "keyword" },
            "journeyPatternId" : { "type": "keyword" },
            "lineId" : { "type": "keyword" },
            "coordinates" : { "type": "geo_point" },
            "timestamp" : { "type": "date", "format" : "epoch_millis" },
            "vehicleId" : { "type": "long" },
            "dateTime" : { "type": "date" },
            "reverse_gecode" : { "type": "keyword" }
        }
    }
}

write_to_elastic(reverse_gecode_df, 'reverse-gecode-index', settings=settings, append=False)


# In[40]:


reverse_gecode_df = read_elastic('reverse-gecode-index', array_field="reverse_gecode")

stop_info_df = read_elastic('stop-information-index').select('stopid', 'shortname')

joined_df = reverse_gecode_df                .join(stop_info_df, reverse_gecode_df['busStop'] == stop_info_df['stopid'], how='inner')                .withColumn('lev_distance', get_text_distance(F.col('shortname'), F.col('reverse_gecode')))                .drop('stopid')


# In[41]:


settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "actualDelay" : { "type": "long" },
            "areaId" : { "type": "long" },
            "areaId1" : { "type": "long" },
            "areaId2" : { "type": "long" },
            "areaId3" : { "type": "long" },
            "atStop" : { "type": "boolean" },
            "busStop" : { "type": "long" },
            "congestion" : { "type": "boolean" },
            "gridID" : { "type": "keyword" },
            "journeyPatternId" : { "type": "keyword" },
            "lineId" : { "type": "keyword" },
            "coordinates" : { "type": "geo_point" },
            "timestamp" : { "type": "date", "format" : "epoch_millis" },
            "vehicleId" : { "type": "long" },
            "dateTime" : { "type": "date" },
            "shortname" : { "type" : "keyword" },
            "reverse_gecode" : { "type": "keyword" },
            "lev_distance" : { "type": "double" }
        }
    }
}

write_to_elastic(joined_df, 'lev-dist-index', settings=settings, append= False)


# ## Filter by Levenshtien:

# In[3]:


stop_df = read_elastic('lev-dist-index', array_field="reverse_gecode")

filter_stop = stop_df.filter("lev_distance < 0.2").drop('lev_distance')

settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "actualDelay" : { "type": "long" },
            "areaId" : { "type": "long" },
            "areaId1" : { "type": "long" },
            "areaId2" : { "type": "long" },
            "areaId3" : { "type": "long" },
            "atStop" : { "type": "boolean" },
            "busStop" : { "type": "long" },
            "congestion" : { "type": "boolean" },
            "gridID" : { "type": "keyword" },
            "journeyPatternId" : { "type": "keyword" },
            "lineId" : { "type": "keyword" },
            "coordinates" : { "type": "geo_point" },
            "timestamp" : { "type": "date", "format" : "epoch_millis" },
            "vehicleId" : { "type": "long" },
            "dateTime" : { "type": "date" },
            "shortname" : { "type" : "keyword" },
            "reverse_gecode" : { "type": "keyword" }
        }
    }
}

write_to_elastic(filter_stop, 'filter-lev-dist-index', settings=settings, append=False)


# ## Calculate Centroids:

# In[6]:


# filtered_df = read_elastic('filter-lev-dist-index')
stop_df = read_elastic('lev-dist-index', array_field="reverse_gecode")

filtered_df = stop_df.filter("lev_distance < 0.5").drop('lev_distance')

filtered_df_centroids = calculate_centroids(filtered_df.select("busStop", "coordinates"))


settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "busStop" : { "type": "long" },
            "coordinates" : { "type": "geo_point" },
        }
    }
}

write_to_elastic(filtered_df_centroids, index="filter-0.5-lev-dist-centroid-index", settings=settings, append=False)


# In[7]:


stop_centroid_df = read_elastic("filter-0.5-lev-dist-centroid-index")
true_coord_df = read_elastic('true-centroid-index')

eval_df = true_coord_df.withColumnRenamed('coordinates', 'true_coordinates')
eval_df = eval_df.join(stop_centroid_df, on='busStop', how='inner').withColumnRenamed('coordinates', 'centroid_coordinates')

mse_centroid = eval_df.agg(F.mean(F.pow(get_distance(eval_df.true_coordinates, eval_df.centroid_coordinates), 2)).alias('mse')).collect()[0]['mse']

print(f"The MSE Value for Centroid is {mse_centroid}")


# ## Analysis 1:

# In[ ]:


stops_df = read_elastic('stop-information-index')
true_coords = read_elastic('true-coord-index').select('busStop')
stops_df = stops_df.join(true_coords, stops_df.stopid == true_coords.busStop, how='inner').drop('stopid')
stops_df = stops_df.select('busStop', 'fullname', 'coordinates').where(stops_df.fullname != '').orderBy('busStop')


# In[ ]:


stops_df1 = stops_df.                withColumnRenamed('busStop', 'busStop1').                withColumnRenamed('fullname', 'fullname1').                withColumnRenamed('coordinates', 'coordinates1')
stops_df2 = stops_df.                withColumnRenamed('busStop', 'busStop2').                withColumnRenamed('fullname', 'fullname2').                withColumnRenamed('coordinates', 'coordinates2')


# In[ ]:


stops_distances = stops_df1.crossJoin(stops_df2)
stops_distances = stops_distances.        withColumn('distance', get_distance(F.col('coordinates1'), F.col('coordinates2'))).        withColumn('text_distance', F.levenshtein(F.col('fullname1'), F.col('fullname2')))
        
w = Window.partitionBy("busStop1")
scaled_result = (F.col("distance") - F.min("distance").over(w)) / (F.max("distance").over(w) - F.min("distance").over(w))
stops_distances = stops_distances.withColumn("scaled_distance", scaled_result).            withColumn("normalized_text_distance", normalize_text_distance(F.col('fullname1'), F.col('fullname2'), F.col('text_distance')))
stops_distances = stops_distances.select('busStop1', 'busStop2', 'text_distance', 'normalized_text_distance', 'distance', 'scaled_distance')


# In[ ]:


distances_index = 'stops-distances-index'
settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "busStop1" : { "type": "long" },
            "busStop2" : { "type": "long" },
            "text_distance" : { "type": "double" },
            "distance" : { "type": "double" },
            "normalized_text_distance" : { "type": "double" },
            "scaled_distance" : { "type": "double" }
        }
    }
}
write_to_elastic(stops_distances, distances_index, settings, append=False)


# In[ ]:


distances_index = 'stops-distances-index'
stop_num = 7207
stop = read_elastic(index=distances_index).where(F.col('busStop1') == stop_num).toPandas()

plt.figure(figsize=(8, 5))
sns.regplot(stop.scaled_distance, stop.normalized_text_distance)
plt.xlabel('Scaled Distance')
plt.ylabel('Normaliazed Levenshtein Distance')
plt.title(f'Stop Number {stop_num}', fontsize=16)
plt.show()


# ## Analysis 2:

# In[ ]:


stops_df = read_elastic('stop-information-index')
true_coords = read_elastic('true-coord-index').select('busStop')
stops_df = stops_df.    join(true_coords, stops_df.stopid == true_coords.busStop, how='inner').    drop('stopid').    where(stops_df.shortnamelocalized != '').    orderBy('busStop')


# In[ ]:


english = stops_df.select('busStop', 'shortname').withColumnRenamed('busStop', 'busStop1')
irish = stops_df.select('busStop', 'shortnamelocalized').withColumnRenamed('busStop', 'busStop2')


# In[ ]:


cross_lang = english.crossJoin(irish)


# In[ ]:


cross_lang = cross_lang.        withColumn('text_distance', F.levenshtein(F.col('shortname'), F.col('shortnamelocalized'))).        withColumn("normalized_text_distance", normalize_text_distance(F.col('shortname'), F.col('shortnamelocalized'), F.col('text_distance')))


# In[ ]:


pivot_cross_lang = cross_lang.groupby('busStop1').pivot('busStop2').agg(F.first('normalized_text_distance'))
pivot_cross_lang = pivot_cross_lang.orderBy('busStop1', ascending=True).drop('busStop1').toPandas()
pivot_cross_lang = 1 - pivot_cross_lang


# # ------------------------------------------------------------------------------------------------

# In[ ]:


# need to remove cell
pivot_cross_lang = 1 - pd.read_pickle('lang.pkl')
num_stops = pivot_cross_lang.shape[0]
best = num_stops * np.diag(pivot_cross_lang) - pivot_cross_lang.sum(axis=1)
s = np.array(best.sort_values(ascending=False).index)
pivot_cross_lang = pivot_cross_lang.iloc[s,s]


# # ------------------------------------------------------------------------------------------------

# In[ ]:


plt.figure(figsize=(8, 5))
sns.heatmap(pivot_cross_lang.iloc[:1001,:1001], cmap=plt.cm.Blues)
plt.xlabel('Irish busStop Name')
plt.ylabel('English busStop Name')
plt.xticks([], [])
plt.yticks([], [])
plt.title('Levenshtein Similarity', fontsize=16)
plt.show()

