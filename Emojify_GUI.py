# Import required Libraries
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd

def login():
    username = entry_username.get()
    password = entry_password.get()

    # Perform login verification logic here
    if username == "admin" and password == "admin":
        messagebox.showinfo("Login Successful", "Welcome to Emojify!")
        root.destroy()
        start_emojify()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

#main code
def start_emojify():

    import tkinter as tk
    from PIL import Image
    from PIL import ImageTk
    import numpy as np
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D

    # Build the convolution network architecture
    face_model = Sequential()
    face_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    face_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    face_model.add(MaxPooling2D(pool_size=(2, 2)))
    face_model.add(Dropout(0.25))
    face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    face_model.add(MaxPooling2D(pool_size=(2, 2)))
    face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    face_model.add(MaxPooling2D(pool_size=(2, 2)))
    face_model.add(Dropout(0.25))
    face_model.add(Flatten())
    face_model.add(Dense(1024, activation='relu'))
    face_model.add(Dropout(0.5))
    face_model.add(Dense(7, activation='softmax'))

   # Load the saved weights
    face_model.load_weights('C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/recognition_model.h5')
#    face_model.load_weights('recognition_model.h5') 

    # Disable OpenCL
    cv2.ocl.setUseOpenCL(False)
    # Create Datasets Dictionaries
    facial_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
    emojis_dict = {0:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/angry.png",
                    1:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/disgusted.png", 
                    2:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/fearful.png", 
                    3:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/happy.png", 
                    4:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/neutral.png", 
                    5:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/sad.png", 
                    6:"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/emojis/surprised.png"}

    # Global variables
    global last_frame1
    last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    global cap1
    show_text=[0]
    emotion_list = []

    # Function to get face captured and recognize emotion
    def Capture_Image():
        global cap1
        cap1 = cv2.VideoCapture(0)
        if not cap1.isOpened():
            print("Can't open the camera1")
        flag1, frame1 = cap1.read()
        frame1 = cv2.resize(frame1, (600, 500))
        # It will detect the face in the video and eliminate the rest objects visible and bound it with a rectangular box
        #bound_box = cv2.CascadeClassifier('C:/Users/onkar/Downloads/Project-Emojify-main/Project-Emojify-main/haarcascades_cuda/haarcascade_frontalface_default.xml') hmm cooll
        bound_box = cv2.CascadeClassifier('C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/haarcascades_cuda/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        n_faces = bound_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in n_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_frame = gray_frame[y:y + h, x:x + w]
            crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
            prediction = face_model.predict(crop_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame1, facial_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex
            emotion_list.append(maxindex)  # Store the detected emotion in the list

        if flag1 is None:
            print("Error!")

        elif flag1:
            global last_frame1
            last_frame1 = frame1.copy()
            pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)  # to store the image
            img = Image.fromarray(pic)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(25, Capture_Image)  # a delay of 25 milliseconds

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    # Function for showing Emoji According to Facial Expression
    def Get_Emoji():
        frame2=cv2.imread(emojis_dict[show_text[0]])
        pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
        img2=Image.fromarray(frame2)
        imgtk2=ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2=imgtk2
        lmain3.configure(text=facial_dict[show_text[0]],font=('arial',45,'bold'))
        lmain2.configure(image=imgtk2)
        lmain2.after(10, Get_Emoji)
    
    # Function to generate a report
    def generate_report():
        import datetime
        import pyautogui
        
        # Get current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        
        # Save screenshot to specified path
        screenshot_path = r"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/op/screenshot_{}.png".format(current_datetime)
        screenshot.save(screenshot_path)
        
        # Generate report content
        report_content = f"Username: {entry_username.get()}\nDate: {current_datetime}\nTime: {current_datetime}\nScreenshot Path: {screenshot_path}"
        
        # Save report to specified path
        report_path = r"C:/Users/shend/OneDrive/Desktop/Emojify/BE final year project/emojify project using DL/Project-Emojify-main/Project-Emojify-main/op/report_{}.txt".format(current_datetime)
        with open(report_path, "w") as file:
            file.write(report_content)
        
        messagebox.showinfo("Report Generated", "Report has been generated successfully.")
    
    def generate_emotion_statistics():
        # Convert the emotion_list to a pandas Series
        emotion_series = pd.Series(emotion_list)

        # Calculate the frequency of each emotion
        emotion_counts = emotion_series.value_counts()

        # Create a bar plot of the emotion frequencies
        plt.figure(figsize=(8, 6))
        emotion_counts.plot(kind='pie')
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')
        plt.title('Emotion Frequency')
        plt.xticks(rotation=45)

        # Calculate the distribution of emotions
        emotion_distribution = emotion_series.value_counts(normalize=True)

        # Set the labels for the pie chart using the emotion names from the facial_dict
        labels = [f"{facial_dict[emotion]}" for emotion, count in zip(emotion_counts.index, emotion_counts.values)]

        # Create a pie chart of the emotion distribution
        plt.figure(figsize=(6, 6))
        plt.pie(emotion_distribution, labels=labels, autopct='%1.1f%%')
        plt.title('Emotion Statictics')
        plt.show()


    # GUI Window to show captured image with emoji
    if __name__ == '__main__':
        root=tk.Tk()
        heading = Label(root,bg='black')
        heading.pack()
        heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')#to label the output
        heading2.pack()
        lmain = tk.Label(master=root,padx=50,bd=10)
        lmain2 = tk.Label(master=root,bd=10)
        lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
        lmain.pack(side=LEFT)
        lmain.place(x=50,y=250)
        lmain3.pack()
        lmain3.place(x=960,y=250)
        lmain2.pack(side=RIGHT)
        lmain2.place(x=900,y=350)
        root.title("Emojify")
        root.geometry("1400x900+100+10")
        root['bg']='black'
        #button_generate_report = tk.Button(root, text="Snapshot", command=generate_report, relief="solid", bg="#16A085", fg="white", font=("Arial", 12, "bold"))
        #button_generate_report.pack(side="top", padx=20, pady=10)

        button_statistics = tk.Button(root, text="Emotion Statistics", command=generate_emotion_statistics, relief="solid", bg="#2980B9", fg="white", font=("Arial", 12, "bold"))
        button_statistics.pack(side="top", padx=20, pady=10)

        Capture_Image()
        Get_Emoji()
        root.mainloop()


# Create the login window
root = tk.Tk()
root.title("Login")
root.geometry("400x300")
root.configure(bg="#2C3E50")

# Create a frame for the logo
logo_frame = tk.Frame(root, bg="#2C3E50")
logo_frame.pack(pady=20)

# Logo
logo_label = tk.Label(logo_frame, text="Emojify", font=("Arial", 20, "bold"), fg="#FFFFFF", bg="#2C3E50")
logo_label.pack()

# Create a frame for the login form
login_frame = tk.Frame(root, bg="#2C3E50")
login_frame.pack(pady=20)

# Username label and entry
label_username = tk.Label(login_frame, text="Username:", bg="#2C3E50", fg="#FFFFFF", font=("Arial", 12))
label_username.grid(row=0, column=0, padx=10, pady=5)

entry_username = tk.Entry(login_frame, font=("Arial", 12), bg="#34495E", fg="#FFFFFF", bd=0)
entry_username.grid(row=0, column=1, padx=10, pady=5)

# Password label and entry
label_password = tk.Label(login_frame, text="Password:", bg="#2C3E50", fg="#FFFFFF", font=("Arial", 12))
label_password.grid(row=1, column=0, padx=10, pady=5)

entry_password = tk.Entry(login_frame, show="*", font=("Arial", 12), bg="#34495E", fg="#FFFFFF", bd=0)
entry_password.grid(row=1, column=1, padx=10, pady=5)

# Login button
button_login = tk.Button(login_frame, text="Login", font=("Arial", 12), bg="#16A085", fg="white", bd=0, padx=20, pady=5, command=login)
button_login.grid(row=2, columnspan=2, pady=10)

root.mainloop()