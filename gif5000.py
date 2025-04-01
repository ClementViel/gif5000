import sys
import shutil
import os
import cv2
# import face_reco
from tkinter import *
import tkinter.filedialog
import tkinter as tk
from PIL import Image, ImageTk
import face_reco


GALLERY = "/Users/clem/Projets/prog/gif5000/photo_tests/"
ANCHOR = "/Users/clem/Projets/prog/gif5000/set/anchor/"


def get_framed_pictures():
    print("filename 1 = "+image1_path)
    if image1_path != "nope" and image2_path != "nope":
        filename1 = image1_path
        filename2 = image2_path
    else:
        filename1 = ""
        filename2 = ""
        print("no image path")
    return filename1, filename2


def write_to_path(image, path):
    num_files = len(os.listdir(path))
    filename = path + "photo" + str(num_files) + ".jpg"
    print("writing " + filename)
    cv2.imwrite(filename, image)


def take_picture():
    print("CLIC")
    s, img = cam.read()
    if s:
        write_to_path(img, GALLERY)


def take_picture_anchor():
    s, img = cam.read()
    if s:
        write_to_path(img, ANCHOR)


def compare_picture():
    # Define function to show frame
    global inference_text
    pic1, pic2 = get_framed_pictures()
    print(pic1 + ", " + pic2 + "loaded")
    distance = face_reco.compare(pic1, pic2)


def load_picture_1():
    global image1_path
    image_file = tk.filedialog.askopenfile(mode='r', initialdir=GALLERY)
    image1_path = image_file.name
    print("Changed" + image1_path)
    img = Image.open(image_file.name).resize((200, 200))
    imgtk = ImageTk.PhotoImage(image=img)
    photo1_label = Label(photo1_frame, image=imgtk)
    photo1_label.image = imgtk
    photo1_label.pack()


def load_picture_2():
    global image2_path
    image_file = tk.filedialog.askopenfile(mode='r', initialdir=GALLERY)
    image2_path = image_file.name
    print("Changed" + image2_path)
    img = Image.open(image_file.name).resize((200, 200))
    imgtk = ImageTk.PhotoImage(image=img)
    photo_label = Label(photo2_frame, image=imgtk)
    photo_label.image = imgtk
    photo_label.pack()


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
    view.after(20, show_frames_camera)


view = Tk()
view.title("GIF5000")
view.geometry("1000x1000")
inference_text = StringVar(name="inference")
inference_text.set("Match ?")
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
label_inference = tk.Label(master=photo_frame, textvariable=inference_text)
label_inference.grid(row=1, column=1)

buttons_frame = tk.Frame(master=view,
                         width=500, height=500, borderwidth=1)
buttons_frame.grid(row=1, column=0)

button_go = tk.Button(master=buttons_frame, text="GO", command=compare_picture)
button_go.grid(row=0, column=0)
button_change1 = tk.Button(
    master=buttons_frame, text="change photo 1", command=load_picture_1)
button_change1.grid(row=1, column=0)
button_change2 = tk.Button(
    master=buttons_frame, text="change photo 2", command=load_picture_2)
button_change2.grid(row=2, column=0)
button_heat = tk.Button(master=buttons_frame, text="view heatmap")
button_heat.grid(row=3, column=0)
button_land = tk.Button(master=buttons_frame, text="view landmark")
button_land.grid(row=4, column=0)
button_photo = tk.Button(master=buttons_frame,
                         text="photo", command=take_picture)
button_photo.grid(row=5, column=0)
button_anchor = tk.Button(master=buttons_frame,
                          text="photo anchor", command=take_picture_anchor)
button_anchor.grid(row=6, column=0)

image1_path = "nope"
image2_path = "nope"
cam = cv2.VideoCapture(0)
show_frames_camera()
view.mainloop()
