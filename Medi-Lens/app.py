
# importing the necessary tools
from flask import Flask, render_template, request,redirect, url_for, session
# to let flask interact easily while performing file and folder processes irrespective of operating system
import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD
import sys
# load the model using joblib
import joblib
import numpy as np
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras_preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# root directory
webroot = 'src'
static_dir = os.path.join(webroot,'static')
template_dir = os.path.join(webroot,'templates')
# defining the flask app
app = Flask(__name__,static_folder=static_dir,template_folder=template_dir)
# route for home page
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

# route for disease section
@app.route('/disease',methods=['GET','POST'])
def disease():
    return render_template('disease.html')

@app.route('/skincancer', methods=["GET", "POST"])
def skincancer():

    return render_template("skin.html")


# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.


@app.route("/showresult", methods=["GET", "POST"])
def show():
    pic = request.files["pic"]
    inputimg = Image.open(pic)
    inputimg = inputimg.resize((28, 28))
    img = np.array(inputimg).reshape(-1, 28, 28, 3)
    result = SCD.model.predict(img)

    result = result.tolist()
    print(result)
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)
    print(class_ind)
    result = SCD.classes[class_ind]

    if class_ind == 0:
        info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion."

    elif class_ind == 1:
        info = "Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off.Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms. Basal cell carcinoma occurs most often on areas of the skin that are exposed to the sun, such as your head and neck"
    elif class_ind == 2:
        info = "Benign lichenoid keratosis (BLK) usually presents as a solitary lesion that occurs predominantly on the trunk and upper extremities in middle-aged women. The pathogenesis of BLK is unclear; however, it has been suggested that BLK may be associated with the inflammatory stage of regressing solar lentigo (SL)1"
    elif class_ind == 3:
        info = "Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. They are firm and often feel like a stone under the skin. "
    elif class_ind == 4:
        info = "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with ‘melanocytic nevus’, but there are also sources that equate the term mole with any nevus form."
    elif class_ind == 5:
        info = "Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color. They tend to bleed because they contain a large number of blood vessels. They’re also known as lobular capillary hemangioma or granuloma telangiectaticum."
    elif class_ind == 6:
        info = "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat. The exact cause of all melanomas isn't clear, but exposure to ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases your risk of developing melanoma."

    return render_template("reults.html", result=result, info=info)






# breast cancer section
@app.route('/breastcancer',methods=['GET','POST'])
def breastcancer():
    if request.method == "POST":
        try:
            Radius_Mean = float(request.form["radius_mean"])
            Area_Mean = float(request.form["area_mean"])
            Compactness_Mean = float(request.form["compactness_mean"])
            Concavity_Mean = float(request.form["concavity_mean"])
            Concave_Points_Mean = float(request.form["concave_points_mean"])
            Area_Worst = float(request.form["area_worst"])
            Compactness_Worst = float(request.form["compactness_worst"])
            Concavity_Worst = float(request.form["concavity_worst"])
            Area_Se = float(request.form["area_se"])
            Fractal_Dimension_Se = float(request.form["fractal_dimension_se"])
            Symmetry_Worst = float(request.form["symmetry_worst"])
            Fractal_Dimension_Worst = float(request.form["fractal_dimension_worst"])

            breast_file = "breast_model.sav"
            loaded_breast_model = joblib.load(breast_file)
            breast_pred = loaded_breast_model.predict([[Radius_Mean, Area_Mean, Compactness_Mean, Concavity_Mean,
            Concave_Points_Mean, Area_Worst, Compactness_Worst,Concavity_Worst, 
            Area_Se, Fractal_Dimension_Se, Symmetry_Worst, Fractal_Dimension_Worst]])
            breast_pred = round(100*breast_pred[0])
            if(breast_pred == 0):
                res = "Congratulations! you are safe from Breast Cancer"
            else:
                res = "Sorry :( you have encountered with Breast Cancer"
            return render_template('breastcancer.html',prediction=res)

        except Exception as e:
            print(e)
            error=("Please fill all the fields for prediction🤦🤦")
            error={"error":error}
            return render_template("404.html",error=error)
    else:
        return render_template('breastcancer.html')

# Diabetes section
@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    if request.method == "POST":
        try:
            Pregnancies = float(request.form["Pregnancies"])
            Glucose = float(request.form["Glucose"])
            Bloodpressure = float(request.form["Bloodpressure"])
            SkinThickness = float(request.form["SkinThickness"])
            Insulin = float(request.form["Insulin"])
            BMIn = float(request.form["BMI"])
            DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
            Age = float(request.form["Age"])
            filename = "diabetes.sav"
            loaded_model = joblib.load(filename)
            dia_pred = loaded_model.predict([[Pregnancies,Glucose,Bloodpressure,SkinThickness,Insulin,BMIn,DiabetesPedigreeFunction,Age]])
            dia_pred = round(100*dia_pred[0])
            if(dia_pred == 0):
                res = "Congratulations! you are safe from Diabetes"
            else:
                res = "Sorry :( you have encountered with Diabetes"
            return render_template('diabetes.html',prediction=res)

        except Exception as e:
            print(e)
            error=("Please fill all the fields for diabetes prediction🤦🤦")
            error={"error":error}
            return render_template("404.html",error=error)
    else:
        return render_template('diabetes.html')

@app.route('/heart',methods=['GET','POST'])
def heart():
    if request.method == 'POST':
        try:
            Age = float(request.form["age"])
            sex = (request.form["sex"])
            if(sex == "male"):
                sex = 1
            else:
                sex = 0
            #chest pain
            chestpain = (request.form["chestpain"])
            if(chestpain == "ATA"):
                chestpain = 1
            elif (chestpain == "NAP"):
                chestpain = 2
            elif (chestpain == "ASY"):
                chestpain = 0
            else:
                chestpain = 3
            # resting bp
            restingbp = float(request.form["restingbp"])
            cholestrol = float(request.form["cholestrol"])
            fastingbs = float(request.form["fastingbs"])
            restingecg = (request.form["restingecg"])
            if(restingecg == "Normal"):
                restingecg = 1
            elif (restingecg == "ST"):
                restingecg = 2
            else:
                restingecg = 0

            maxhr = float(request.form["maxhr"])
            exercise = (request.form["exercise"])
            if(exercise == "N"):
                exercise = 0
            else:
                exercise = 1

            oldpeak = float(request.form["oldpeak"])
            stslope = (request.form["stslope"])
            if(stslope == "up"):
                stslope = 2
            elif(stslope == "flat"):
                stslope = 1
            else:
                stslope = 0
            file_heart = "heart_model.sav"
            loaded_model = joblib.load(file_heart)
            heart_pred = loaded_model.predict([[Age, sex, chestpain, restingbp, cholestrol, fastingbs, restingecg, maxhr,
            exercise, oldpeak, stslope]])

            heart_pred = round(100*heart_pred[0])
            if(heart_pred == 0):
                res = "Congratulations! you are safe from Heart Disease"
            else:
                res = "Sorry :( you have encountered with Heart Failure"
            return render_template('heart.html',predict = res)

        except Exception as e:
            print(e)
            error=("Please fill all the fields for heart disease prediction🤦🤦")
            error={"error":error}
            return render_template("404.html",error=error)
    else:
        return render_template('heart.html')

#pneumonia prediction section

# load pneumonia model path
PNEUMONIA_MODEL_PATH = 'pneumonia_model.h5'

#Load your trained model
model = load_model(PNEUMONIA_MODEL_PATH)
# pneumonia detection
def pneumonia_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64)) #target_size must agree with what the trained model expects!!
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return preds

@app.route('/pneumonia',methods=['GET','POST'])
def pneumonia():
    if request.method == 'POST':
        # try:
            # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, '', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = pneumonia_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned
        if preds == 1:
            res = "Sorry :( you have got the chances of Pneumonia"
        else:
            res = "Congratulations! you are safe from Pneumonia"
        return render_template('pneumonia.html',prediction=res)
        # except Exception as e:
        #     print(e)
        #     error=("Have you uploaded the image??🤔🤔")
        #     error={"error":error}
        #     return render_template("404.html",error=error)
    return render_template('pneumonia.html')

# thyroid static page
@app.route('/thyroid',methods=['GET','POST'])
def thyroid():
    return render_template('thyroid.html')
# Driver code
port = int(os.environ.get("PORT", 5000))

if __name__=="__main__":
    
    app.run(debug=True,port=port,host="0.0.0.0")