import numpy as np
import pandas as pd
import cv2
import redis
import os
from dotenv import load_dotenv

import av
import tensorflow as tf
import pickle

load_dotenv()

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

model_path='liveness.model'
le_path='label_encoder.pickle'
encodings='encoded_faces.pickle'
detector_folder='face_detector'
confidence=0.5
args = {'model':model_path, 'le':le_path, 'detector':detector_folder,
	'encodings':encodings, 'confidence':confidence}

with open(args['encodings'], 'rb') as file:
	encoded_data = pickle.loads(file.read())

# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())


# Connect to Redis
hostname = 'redis-13570.c305.ap-south-1-1.ec2.cloud.redislabs.com'
port = '13570'
password = 'wMDKrOv3lzrwQUkJV16DXKR3Jble9V4l'

r = redis.Redis(host=hostname, port=port, password=password)

# retrieve data from Redis
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode('utf-8'), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name', 'facial_features']
    return retrive_df[['name', 'facial_features']]

# configure face analysis
app = FaceAnalysis(name="buffalo_sc", root="insightface_model", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, thresh=0.5):
    """
    Cosine similarity search algorithm
    """
    # 1. take dataframe (collection of all data)
    dataframe = dataframe.copy()

    # 2. index face embedding from dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    X = np.array(X_list)

    # 3. Calculate cosine similarity between face embedding from dataframe and face embedding from test image
    similarity = pairwise.cosine_similarity(X, test_vector.reshape(1, -1))
    similarity_arr = np.array(similarity).flatten()
    dataframe["cosine_similarity"] = similarity_arr

    # 4. filter the dataframe based on cosine similarity
    data_filter = dataframe.query("cosine_similarity >= @thresh")
    if len(data_filter) > 0:
        # 5. get the most similar person from filtered dataframe
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter["cosine_similarity"].argmax()
        name = data_filter.loc[argmax]["name"]
    else:
        name = "Unknown"
    return name

# Realtime Prediction
# save logs for every 1 min
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[])


    def face_prediction(self, test_image, dataframe, feature_column, thresh=0.5):
        name = 'Unknown'
        y1, y2, x1, x2 = 0, 0, 0, 0

        # 1. take test image and apply face detection using insightface
        results = app.get(test_image)
        test_copy = test_image.copy()

        # 2. use for loop and extract face embedding from each face detected and pass it to ml_search_algorithm function
        for i, res in enumerate(results):
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embedding = res['embedding']
            name = ml_search_algorithm(dataframe, feature_column, test_vector=embedding, thresh=thresh)

            # 3. perform liveness detection using the same bounding box
            face = test_copy[y1:y2, x1:x2]
            try:
                face = cv2.resize(face, (32, 32))
            except:
                break

            face = face.astype('float') / 255.0
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = liveness_model.predict(face)[0]
            j = np.argmax(preds)
            label_name = le.classes_[j]  # get label of predicted class

            # check if the predicted class is "fake" and the probability threshold is beyond 0.9
            if label_name == 'fake' and preds[j] > 0.9:
                cv2.putText(test_copy, "Not Live", (x1, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

        if name == 'Unknown':
            color = (0, 0, 255)  # bgr
        else:
            color = (0, 255, 0)

        cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

        cv2.putText(test_copy, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        # save logs
        self.logs['name'].append(name)

        return test_copy

##### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        y1, y2, x1, x2 = 0, 0, 0, 0
        # get result from face analysis
        results = app.get(frame, max_num=1)  # max_num: maximum number of faces to detect
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 2: thickness
            cv2.putText(frame, 'samples =' + ' ' + str(self.sample), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (0, 255, 0), 2)

            # get facial features
            embeddings = res['embedding']
        return frame, embeddings

    def save_data_in_redis_db(self, name):
        if name is not None:
            if name.strip() != '':
                key = name
            else:
                return False
        else:
            return 'Name cannot be empty'

        if 'face_embedding.txt' not in os.listdir():
            return False

        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # save to redis
        r.hset('academy:register', key=key, value=x_mean_bytes)

        os.remove('face_embedding.txt')

        self.reset()
        return True
