import sys
import shutil
import os
import cv2
import face_reco
from tkinter import *
from PIL import Image, ImageTk

GALLERY = "/Users/clem/Projets/prog/gif5000/photo_tests/"


def get_two_last_pictures():
    num_files = len(os.listdir(GALLERY))
    filename1 = "null"
    filename2 = "null"

    if num_files >= 2:
        filename1 = GALLERY + "photo" + str(num_files - 1) + ".jpg"
        filename2 = GALLERY + "photo" + str(num_files - 2) + ".jpg"

    return filename1, filename2


def write_to_gallery(image):
    num_files = len(os.listdir(GALLERY))
    filename = GALLERY + "photo" + str(num_files) + ".jpg"
    print("writing " + filename)
    cv2.imwrite(filename, image)


def take_picture():
    s, img = cam.read()
    if s:
        cv2.namedWindow("cam-test")
        cv2.imshow("cam-test", img)
        cv2.waitKey(0)
        cv2.destroyWindow("cam-test")
        write_to_gallery(img)


def compare_picture():
    # Define function to show frame
    pic1, pic2 = get_two_last_pictures()
    print(pic1 + ", " + pic2 + "loaded")
    face_reco.compare(pic1, pic2)


def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.grid(row=0, column=0)
    # Repeat after an interval to capture continiously
    label.after(20, show_frames)


view = Tk()
label = Label(view)
label.grid(row=3, column=2)
photo_button = Button(view, text="Photo !", command=take_picture)
compare_button = Button(view, text="Compare !", command=compare_picture)
photo_button.grid(row=1, column=0)
compare_button.grid(row=1, column=1)
cam = cv2.VideoCapture(0)
show_frames()
view.mainloop()
