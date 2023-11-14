import argparse
import pymysql
import logging
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# database information
endpoint = ''
username = ''
password = ''
database_name = ''

# in the intial pull, there are approximately 1100 racers (i.e. identities that we care about)


def main(num_clusters, write_to_db=False, write_to_file=False):

    logger.debug(f'getting all embeddings from database')

    # now get the event embeddings, and the biggest bbox
    with pymysql.connect(host=endpoint, user=username, password=password, database=database_name) as connection:
        with connection.cursor() as cursor:
            # Then, fetch the id and embeddings for all detected_objects with the given label_stage
            # sql = """SELECT id, image_id, group_id, s3url, label_stage, embedding FROM detected_objects WHERE label_stage = 5"""
            # cursor.execute(sql, (1,))
            # rows = cursor.fetchall()

            # fetch all column names from the detected_objects table
            # sql = """SELECT id, image_id, group_id, s3url, label_stage, embedding FROM detected_objects"""
            sql = """SHOW COLUMNS FROM detected_objects"""
            # sql = """SELECT * FROM detected_objects WHERE label_stage = %s"""
            # #count the number of unique group_ids in the detected_objects table, and print formatted output to console
            # sql = """SELECT COUNT(DISTINCT group_id) FROM detected_objects WHERE label_stage = %s"""
            #return counts by group_id of the items in the detected_objects table, ordered by the count
            # sql = """SELECT group_id, COUNT(*) FROM detected_objects GROUP BY group_id ORDER BY COUNT(*) DESC"""
            # sql = """SELECT id, group_id, group_center, s3url, label_stage FROM detected_objects WHERE group_id = 337"""
            cursor.execute(sql)
            rows = cursor.fetchall()
            #count the number of items in rows where row[1] < 5, and where row[1] >= 5
            print(f'number of items in rows where row[1] < 5: {len([row for row in rows if row[1] < 5])}')
            print(f'number of items in rows where row[1] >= 5: {len([row for row in rows if row[1] >= 5])}')
            # print the max value of row[1] and min value of row[1]
            print(f'max value of row[1]: {max([row[1] for row in rows])}')
            print(f'min value of row[1]: {min([row[1] for row in rows])}')

            for row in rows:
                print(f'{row[0]}_{row[1]}')


    # create an empty list or array to store embeddings as numpy arrays
    embeddings = []
    embedding_ids = []

    logger.debug(f'building the numpy array and event info for k = {num_clusters}')

    # iterate over rows and convert blob values to numpy arrays
    for row in rows:
        det_obj_id, embedding = row
        arr = np.frombuffer(embedding, dtype=np.float32)  # convert blob to numpy array of float32 type
        # logger.debug('embedding shape: ', arr.shape)
        embeddings.append(arr)  # append numpy array to list or array
        embedding_ids.append(det_obj_id)

    embeddings = np.array(embeddings)  # convert list or array to numpy array

    logger.debug('=============================================')
    logger.debug('number of clusters: ', num_clusters)
    logger.debug(f'embeddings shape: {embeddings.shape}')
    logger.debug('=============================================')
    # Perform k-means clustering on the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Combine embedding_ids, embeddings, and cluster_labels
    id_embedding_cluster = list(zip(embedding_ids, embeddings, cluster_labels))

    # Find the closest object to the cluster center for each cluster
    nearest_objects = {}
    for obj_id, emb, cluster_label in id_embedding_cluster:
        cluster_center = cluster_centers[cluster_label]
        dist = distance.euclidean(emb, cluster_center)

        if cluster_label not in nearest_objects or dist < nearest_objects[cluster_label][1]:
            nearest_objects[cluster_label] = (obj_id, dist)



    return

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--num_clusters', type=int, default=1000)
    args = argparse.parse_args()
    main(args.num_clusters)
