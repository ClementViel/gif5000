import sys
import shutil
import os
import cv2
# import face_reco
import tkinter as tk
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


def show_frames_camera():
    # Get the latest frame and convert into Image
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        label_camera_frame.photo = imgtk
        label_camera_frame.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    view.after(20, show_frames)


view = Tk()
view.title("GIF5000")
view.geometry("1000x1000")
camera_frame = tk.Frame(
    view, relief=tk.RIDGE, borderwidth=1, width=500, height=500)
camera_frame.pack()
label_camera_frame = tk.Label(master=camera_frame, text="COUCOU")
camera_frame.pack_propagate(False)
label_camera_frame.pack(fill=tk.BOTH, expand=True)
camera_frame.grid(row=0, column=1)

debug_frame = tk.Frame(master=view, relief=tk.RIDGE,
                       borderwidth=1, width=500, height=500)
debug_frame.pack_propagate(False)
debug_frame.grid(row=1, column=1)
label_debug_frame = tk.Label(master=debug_frame, text="debug frame")
label_debug_frame.pack()

photo_frame = tk.Frame(view, relief=tk.RIDGE,
                       borderwidth=1, width=500, height=500)
photo_frame.pack_propagate(False)
label_photo_frame = tk.Label(master=photo_frame, text="Photos et boutons")
label_photo_frame.pack()
photo_frame.grid(row=0, column=0)
photo1_frame = tk.Frame(master=photo_frame, relief=tk.RIDGE,
                        borderwidth=1, width=200, height=200)
photo1_frame.grid(row=0, column=0)
photo1_frame.pack_propagate(False)
photo2_frame = tk.Frame(master=photo_frame, relief=tk.RIDGE,
                        borderwidth=1, width=200, height=200)
photo2_frame.pack_propagate(False)
photo2_frame.grid(row=0, column=1)
label_inference = tk.Label(master=photo_frame, text="Match ?")
label_inference.grid(row=1, column=1)

buttons_frame = tk.Frame(master=view,
                         width=500, height=500, borderwidth=1)
buttons_frame.grid(row=1, column=0)

button_go = tk.Button(master=buttons_frame, text="GO")
button_go.grid(row=0, column=0)
button_change1 = tk.Button(master=buttons_frame, text="change photo 1")
button_change1.grid(row=1, column=0)
button_change2 = tk.Button(master=buttons_frame, text="change photo 2")
button_change2.grid(row=2, column=0)
button_heat = tk.Button(master=buttons_frame, text="view heatmap")
button_heat.grid(row=3, column=0)
button_land = tk.Button(master=buttons_frame, text="view landmark")
button_land.grid(row=4, column=0)


cam = cv2.VideoCapture(0)
show_frames_camera()
view.mainloop()
