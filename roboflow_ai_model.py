
from roboflow import Roboflow
rf = Roboflow(api_key="**********")
project = rf.workspace().project("papia-mouse")
model = project.version(1).model

model.predict("C:\\Users\\HUAWEI\\Desktop\\test\\8.jpg", confidence=40, overlap=30).save("prediction.jpg")

# Roboflow uygulaması, YoloV8 modeli ile nesne tanıma projesi.

 
