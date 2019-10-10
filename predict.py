import sys
import cv2
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

project_id = 'sandbox-nguyenvietcuong'
model_NP_id = 'TCN2036296203375337149'
model_Obj_id= 'IOD4129312675136012288'
project_obj_id='1093967521155'

def get_prediction_NP(content, project_id, model_NP_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_NP_id)
    payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned

def get_prediction_Object_detection(file_path, project_obj_id, model_Obj_id):
    object_coordinate=[]
    with open(file_path, 'rb') as ff:
        content = ff.read()
    img=cv2.imread(file_path)
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_Obj_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    #print(request)
    for item in request.payload:
        label=item.display_name
        score=item.image_object_detection.score
        xmin=item.image_object_detection.bounding_box.normalized_vertices[0].x*img.shape[1]
        ymin=item.image_object_detection.bounding_box.normalized_vertices[0].y*img.shape[0]
        xmax=item.image_object_detection.bounding_box.normalized_vertices[1].x*img.shape[1]
        ymax=item.image_object_detection.bounding_box.normalized_vertices[1].y*img.shape[0]
        object_coordinate.append([label,xmin,ymin,xmax,ymax,score])
        #cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),1)
    print(object_coordinate)
    return object_coordinate  # waits till request is returned
  
  
if __name__ == '__main__':
    img=r"C:\Users\Nguyen Viet CUONG\Desktop\labGCP\readbill\image\page12.jpg"
    
    content = '税抜金額額50,000円費税額'
    value=get_prediction_NP(content, project_id,  model_NP_id)
    result=get_prediction_Object_detection(img,project_obj_id,model_Obj_id)
    #for item in result.payload:
        #print(item.image_object_detection)
    #print(result)
    #for item in value.payload:
        #print(item.classification)
    #print (value.payload)