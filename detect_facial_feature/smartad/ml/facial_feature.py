# Import required modules
import cv2 as cv
import numpy as np
from .beard import detect_beard
from PIL import Image

def predict(file_path):
    from keras.models import load_model
    model=load_model('ml/Age_Sex_Detection.h5')
    global label_packed
    image=Image.open(file_path)
    image=image.resize((48,48))
    image=np.expand_dims(image,axis=0)
    image=np.array(image)
    image=np.delete(image,0,1)
    image=np.resize(image,(48,48,3))
    print(image.shape)
    sex_f=["Male","Female"]
    image=np.array([image])/255
    pred=model.predict(image)
    age=int(np.round(pred[1][0]))
    sex=int(np.round(pred[0][0]))
    sex = sex_f[sex]
    beard = detect_beard(file_path)
    print("Predicted Age is "+ str(age))
    print("Predicted Gender is "+ sex)
    print("Predicted beard is "+ beard)
    res = {'gender': sex, 'age': age, 'beard': beard}
    return res
    #    cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    #    cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
#print(predict('image.jpg'))
