import pandas as pd
import redis
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import warnings
import time

load_dotenv()

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

from src.anti_spoof_predict_pro import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

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

model_test = AntiSpoofPredict(0, "./resources/anti_spoof_models")
image_cropper = CropImage()
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


            # 3. Spoof
            prediction = np.zeros((1, 3))
            for model_name in os.listdir("./resources/anti_spoof_models"):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": test_copy,
                    "bbox": res['bbox'],
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)

                predictions = model_test.predict_batch([img])

                prediction += predictions[model_name]

            label = np.argmax(prediction)
            value = prediction[0][label] / 2
            if label == 1:
                result_text = "Live: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                result_text = "Fake: {:.2f}".format(value)
                color = (0, 0, 255)

            cv2.putText(
                test_copy,
                result_text,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)



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
