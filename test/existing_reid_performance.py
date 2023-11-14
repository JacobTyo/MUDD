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


def main(num_clusters):

    logger.debug(f'getting all embeddings from database')

    # now get the event embeddings, and the biggest bbox
    with pymysql.connect(host=endpoint, user=username, password=password, database=database_name) as connection:
        with connection.cursor() as cursor:
            # Then, fetch the id and embeddings for all detected_objects with the given label_stage
            sql = """SELECT id, image_id, embedding FROM detected_objects"""
            cursor.execute(sql)
            rows = cursor.fetchall()

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
