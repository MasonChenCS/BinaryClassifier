import os
import re
import shutil
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tkinter import *
import tkinter.filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np


def alphanumeric_sort(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r'(\d+)', key)]
    return sorted(data, key=alphanum_key)


def create_model():
    global base_dir
    global universal_size
    global desired_accuracy
    global total_epochs
    global train_batch_size
    global test_batch_size
    global dropout_rate

    absent_dir = base_dir + '/' + 'absent'
    present_dir = base_dir + '/' + 'present'

    if (os.path.exists(absent_dir) and len(os.listdir(absent_dir)) > 0 and os.path.exists(present_dir) and len(
            os.listdir(present_dir)) > 0) or (os.path.exists(os.path.join(base_dir, 'train')) and
                                              os.path.exists(os.path.join(base_dir, 'test'))):
        listbox_console.delete(1, END)
        listbox_console.insert('end', 'Organizing Folders...')
        train_datagenerator = ImageDataGenerator(rescale=1. / 255)
        test_datagenerator = ImageDataGenerator(rescale=1. / 255)

        if os.path.exists(os.path.join(base_dir, 'train/absent')) or os.path.exists(
                os.path.join(base_dir, 'train/present')) or \
                os.path.exists(os.path.join(base_dir, 'test/absent')) or \
                os.path.exists(os.path.join(base_dir, 'test/present')):
            for file_name in os.listdir(os.path.join(base_dir, 'test/absent')):
                shutil.move(os.path.join(base_dir, 'test/absent', file_name), absent_dir)
            for file_name in os.listdir(os.path.join(base_dir, 'train/absent')):
                shutil.move(os.path.join(base_dir, 'train/absent', file_name), absent_dir)
            for file_name in os.listdir(os.path.join(base_dir, 'test/present')):
                shutil.move(os.path.join(base_dir, 'test/present', file_name), present_dir)
            for file_name in os.listdir(os.path.join(base_dir, 'train/present')):
                shutil.move(os.path.join(base_dir, 'train/present', file_name), present_dir)
            shutil.rmtree(os.path.join(base_dir, 'train'))
            shutil.rmtree(os.path.join(base_dir, 'test'))

        absent_count = len(os.listdir(absent_dir))
        present_count = len(os.listdir(present_dir))

        # Create train
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
        # Create train/absent
        train_absent_dir = os.path.join(train_dir, 'absent')
        os.mkdir(train_absent_dir)
        # Create train/present
        train_present_dir = os.path.join(train_dir, 'present')
        os.mkdir(train_present_dir)
        # Create test
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)
        # Create test/absent
        test_absent_dir = os.path.join(test_dir, 'absent')
        os.mkdir(test_absent_dir)
        # Create test/present
        test_present_dir = os.path.join(test_dir, 'present')
        os.mkdir(test_present_dir)
        # Move random 0.2 of absent to test/absent
        for file_name in random.sample(os.listdir(absent_dir), int(0.2 * absent_count)):
            shutil.move(os.path.join(absent_dir, file_name), test_absent_dir)
        # Move remainder of absent to train/absent
        for file_name in os.listdir(absent_dir):
            shutil.move(os.path.join(absent_dir, file_name), train_absent_dir)
        # Move random 0.2 of present to test/present
        for file_name in random.sample(os.listdir(present_dir), int(0.2 * present_count)):
            shutil.move(os.path.join(present_dir, file_name), test_present_dir)
        # Move remainder of present to train/present
        for file_name in os.listdir(present_dir):
            shutil.move(os.path.join(present_dir, file_name), train_present_dir)

        listbox_console.insert('end', 'Starting training...')

        train_datagenerator = train_datagenerator.flow_from_directory(
            base_dir + '/train',
            target_size=(universal_size, universal_size),
            batch_size=train_batch_size,
            class_mode='binary')

        test_datagenerator = test_datagenerator.flow_from_directory(
            base_dir + '/test',
            target_size=(universal_size, universal_size),
            batch_size=test_batch_size,
            class_mode='binary')

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                   input_shape=(universal_size, universal_size, 3)),
            tf.keras.layers.MaxPooling2D((2, 2), 2),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), 2),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

        class MyCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('accuracy') > desired_accuracy and logs.get('val_accuracy') > desired_accuracy:
                    # If the above line doesn't work, try changing 'acc' to 'accuracy' and vice versa
                    # print("\nReached " + str(int(desired_accuracy * 100)) + "% accuracy so cancelling training!")
                    listbox_console.insert('end', '\nReached ' + str(round(desired_accuracy * 100.0, 2)) + '% accuracy. Canceling training!')
                    self.model.stop_training = True

        callbacks = MyCallback()
        model.fit(
            train_datagenerator,
            epochs=total_epochs,
            validation_data=test_datagenerator,
            callbacks=[callbacks],
            verbose=1
        )

        listbox_console.insert('end', 'Finished training. Saving model...')
        model.save('Model.h5')
        tf.keras.backend.clear_session()
        # Move test/absent to absent
        for file_name in os.listdir(test_absent_dir):
            shutil.move(os.path.join(test_absent_dir, file_name), absent_dir)
        # Move train/absent to absent
        for file_name in os.listdir(train_absent_dir):
            shutil.move(os.path.join(train_absent_dir, file_name), absent_dir)
        # Move test/present to present
        for file_name in os.listdir(test_present_dir):
            shutil.move(os.path.join(test_present_dir, file_name), present_dir)
        # Move train/present to present
        for file_name in os.listdir(train_present_dir):
            shutil.move(os.path.join(train_present_dir, file_name), present_dir)
        # Delete train
        shutil.rmtree(train_dir)
        # Delete test
        shutil.rmtree(test_dir)
        listbox_console.insert('end', 'Files moved to original locations. Process is complete!')
    else:
        listbox_console.insert('end', 'Please ensure that there are files for groups A and B.')


def predict():
    if src_predict_dir:
        prepend_dir = False
        listbox_predict.delete(1, END)
        list = []
        if model != '':
            if os.path.isdir(src_predict_dir[0]):
                filelist = [files for files in
                            (alphanumeric_sort(os.listdir(src_predict_dir[0])) if sort_list else os.listdir(src_predict_dir[0]))]
                list = {ind: name for (ind, name) in enumerate(filelist)}
                prepend_dir = True
            elif os.path.isfile(src_predict_dir[0]):
                list = src_predict_dir
            for i in range(len(list)):
                img = cv2.imread((os.path.join(src_predict_dir[0], list[i])) if prepend_dir else list[i], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (universal_size, universal_size))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array, verbose=0)[0][0]
                if prediction < 0.5:
                    listbox_predict.insert('end', '(A) ' + ((list[i]) if prepend_dir else list[i][list[i].rfind("/") + 1:]))
                elif prediction > 0.5:
                    listbox_predict.insert('end', '(B) ' + ((list[i]) if prepend_dir else list[i][list[i].rfind("/") + 1:]))
            tf.keras.backend.clear_session()
        else:
            listbox_predict.insert('end', 'Model not selected.')


def copy_files(mode_):
    global src_absent_dir
    global src_present_dir
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    absent_dir = base_dir + '/' + 'absent'
    present_dir = base_dir + '/' + 'present'
    if mode_ == 'absent':
        listbox_absent.delete(1, END)
        if src_absent_dir:
            if not os.path.exists(absent_dir):
                os.mkdir(absent_dir)
            if os.path.isdir(src_absent_dir[0]):
                for filename in os.listdir(src_absent_dir[0]):
                    shutil.copy(os.path.join(src_absent_dir[0], filename), absent_dir)
            elif os.path.isfile(src_absent_dir[0]):
                for filename in src_absent_dir:
                    shutil.copy(filename, absent_dir)
            listbox_absent.insert('end', 'Files for Group A copied.')
        else:
            listbox_absent.insert('end', 'No files found for Group A.')
    elif mode_ == 'present':
        listbox_present.delete(1, END)
        if src_present_dir:
            if not os.path.exists(present_dir):
                os.mkdir(present_dir)
            if os.path.isdir(src_present_dir[0]):
                for filename in os.listdir(src_present_dir[0]):
                    shutil.copy(os.path.join(src_present_dir[0], filename), present_dir)
            elif os.path.isfile(src_present_dir[0]):
                for filename in src_present_dir:
                    shutil.copy(filename, present_dir)
            listbox_present.insert('end', 'Files for Group B copied.')
        else:
            listbox_present.insert('end', 'No files found for Group B.')


def delete_files():
    global base_dir
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        listbox_console.insert('end', 'Training files deleted.')


def drop_listbox_absent(event):
    global listbox_absent
    update_listbox(event.data, listbox_absent, 'absent')


def drop_listbox_present(event):
    global listbox_present
    update_listbox(event.data, listbox_present, 'present')


def drop_listbox_model(event):
    global listbox_model
    update_listbox(event.data, listbox_model, 'model')


def drop_listbox_predict(event):
    global listbox_predict
    update_listbox(event.data, listbox_predict, 'predict')


def update_listbox(entry, listbox_, mode_):
    global model
    global src_absent_dir
    global src_present_dir
    global src_predict_dir
    if entry != '':
        if entry[0] == '{':
            entry = entry[1:]
        if entry[-1] == '}':
            entry = entry[:-1]
        if os.path.isdir(entry):
            if mode_ == 'absent':
                listbox_absent.delete(1, END)
                listbox_absent.insert('end', 'Chosen directory: ' + entry[entry.rfind("/") + 1:])
                src_absent_dir = [entry]
            elif mode_ == 'present':
                listbox_present.delete(1, END)
                listbox_present.insert('end', 'Chosen directory: ' + entry[entry.rfind("/") + 1:])
                src_present_dir = [entry]
            elif mode_ == 'predict':
                listbox_predict.delete(1, END)
                listbox_predict.insert('end', 'Chosen directory: ' + entry[entry.rfind("/") + 1:])
                src_predict_dir = [entry]
                predict()
        else:
            list = re.split(r"} {|} | {", entry)
            if mode_ == 'absent':
                listbox_absent.delete(1, END)
                listbox_absent.insert('end', 'Chosen files:')
                for item in list:
                    if os.path.isfile(item):
                        listbox_absent.insert('end', item[item.rfind("/") + 1:])
                src_absent_dir = list
            elif mode_ == 'present':
                listbox_present.delete(1, END)
                listbox_present.insert('end', 'Chosen files:')
                for item in list:
                    if os.path.isfile(item):
                        listbox_present.insert('end', item[item.rfind("/") + 1:])
                src_present_dir = list
            elif mode_ == 'model':
                if os.path.isfile(list[0]) and os.path.splitext(list[0])[1] in ['.h5']:
                    listbox_model.delete(1, END)
                    listbox_model.insert('end', 'Using model ' + list[0][list[0].rfind("/") + 1:])
                    model = tf.keras.models.load_model(list[0])
            elif mode_ == 'predict':
                listbox_predict.delete(1, END)
                listbox_predict.insert('end', 'Chosen files:')
                for item in list:
                    if os.path.isfile(item):
                        listbox_predict.insert('end', item[item.rfind("/") + 1:])
                src_predict_dir = list
                predict()


def switch_mode(mode_):
    global mode
    if mode_ == '':
        mode = 'train'
        switch_mode(mode)
    elif mode_ == 'train':
        mode = 'predict'
        listbox_model.grid_forget()
        listbox_predict.grid_forget()
        listbox_absent.grid(row=1, column=0, columnspan=2)
        listbox_present.grid(row=1, column=2, columnspan=2)
        listbox_console.grid(row=4, column=0, columnspan=4)
        button_create_model.grid(row=3, column=0, columnspan=4)
        button_copy_absent.grid(row=2, column=0, columnspan=2)
        button_copy_present.grid(row=2, column=2, columnspan=2)
        button_delete.grid(row=5, column=0, columnspan=4)
        button_settings.grid(row=0, column=3)
        button_switch_mode.configure(width=21)
        button_switch_mode.grid_configure(columnspan=3)
    elif mode_ == 'predict':
        mode = 'train'
        listbox_absent.grid_forget()
        listbox_present.grid_forget()
        listbox_console.grid_forget()
        button_create_model.grid_forget()
        button_copy_absent.grid_forget()
        button_copy_present.grid_forget()
        button_delete.grid_forget()
        button_settings.grid_forget()
        listbox_model.grid(row=1, column=0, columnspan=4)
        listbox_predict.grid(row=2, column=0, columnspan=4)
        button_switch_mode.configure(width=29)
        button_switch_mode.grid_configure(columnspan=4)
    else:
        print('Error selecting mode.')


def open_settings():
    global window
    global settings_window
    global universal_size
    global desired_accuracy
    global total_epochs
    global train_batch_size
    global test_batch_size
    global dropout_rate
    width_ = 393
    height_ = 393 + 21
    label_font_size = 10
    settings_window = Toplevel(window)
    settings_window.title("Binary Classifier Settings")
    settings_window.configure(background=bg_color)
    temp_x = window.winfo_x() + window.winfo_width()
    if window.winfo_x() + window.winfo_width() + width_ >= window.winfo_screenwidth():
        temp_x = window.winfo_x() - width_
    settings_window.geometry("%dx%d+%d+%d" % (
        width_, height_, temp_x, window.winfo_y()))
    settings_window.resizable(False, False)
    button_settings.configure(state=DISABLED)
    button_switch_mode.configure(state=DISABLED)
    button_create_model.configure(state=DISABLED)

    def close_settings():
        global universal_size
        global desired_accuracy
        global total_epochs
        global train_batch_size
        global test_batch_size
        global dropout_rate
        universal_size = image_dim_slider.get()
        desired_accuracy = accuracy_slider.get()
        total_epochs = epochs_slider.get()
        train_batch_size = train_batch_slider.get()
        test_batch_size = test_batch_slider.get()
        dropout_rate = dropout_slider.get()
        button_settings.configure(state=NORMAL)
        button_switch_mode.configure(state=NORMAL)
        button_create_model.configure(state=NORMAL)
        settings_window.destroy()

    settings_window.protocol("WM_DELETE_WINDOW", close_settings)
    image_dim_label = Label(settings_window, text="Image Dimensions", font=("Arial", label_font_size, "bold"),
                            fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    image_dim_label.grid(row=0, column=0)
    image_dim_slider = Scale(settings_window, from_=16, to=1024, resolution=16, orient=HORIZONTAL,
                             font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                             highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                             borderwidth=0)
    image_dim_slider.set(universal_size)
    image_dim_slider.grid(row=1, column=0)
    accuracy_label = Label(settings_window, text="Desired Accuracy", font=("Arial", label_font_size, "bold"),
                           fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    accuracy_label.grid(row=2, column=0)
    accuracy_slider = Scale(settings_window, from_=0.001, to=1, resolution=0.001, orient=HORIZONTAL,
                            font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                            highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                            borderwidth=0)
    accuracy_slider.set(desired_accuracy)
    accuracy_slider.grid(row=3, column=0)
    epochs_label = Label(settings_window, text="Total Epochs", font=("Arial", label_font_size, "bold"),
                         fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    epochs_label.grid(row=4, column=0)
    epochs_slider = Scale(settings_window, from_=1, to=100, resolution=1, orient=HORIZONTAL,
                          font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                          highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                          borderwidth=0)
    epochs_slider.set(total_epochs)
    epochs_slider.grid(row=5, column=0)
    train_batch_label = Label(settings_window, text="Train Batch Size", font=("Arial", label_font_size, "bold"),
                              fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    train_batch_label.grid(row=6, column=0)
    train_batch_slider = Scale(settings_window, from_=1, to=500, resolution=1, orient=HORIZONTAL,
                               font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                               highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                               borderwidth=0)
    train_batch_slider.set(train_batch_size)
    train_batch_slider.grid(row=7, column=0)
    test_batch_label = Label(settings_window, text="Test Batch Size", font=("Arial", label_font_size, "bold"),
                             fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    test_batch_label.grid(row=8, column=0)
    test_batch_slider = Scale(settings_window, from_=1, to=500, resolution=1, orient=HORIZONTAL,
                              font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                              highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                              borderwidth=0)
    test_batch_slider.set(test_batch_size)
    test_batch_slider.grid(row=9, column=0)
    dropout_label = Label(settings_window, text="Dropout Rate", font=("Arial", label_font_size, "bold"),
                          fg="white", width=49, height=2, highlightthickness=0, bg=button_color2, borderwidth=0)
    dropout_label.grid(row=10, column=0)
    dropout_slider = Scale(settings_window, from_=0, to=1, resolution=0.05, orient=HORIZONTAL,
                           font=("Arial", label_font_size, "bold"), fg="white", length=width_, sliderlength=20,
                           highlightthickness=0, bg=button_color2, troughcolor=slider_color,
                           borderwidth=0)
    dropout_slider.set(dropout_rate)
    dropout_slider.grid(row=11, column=0)


sort_list = True
universal_size = 64
desired_accuracy = 0.999
total_epochs = 15
train_batch_size = 40
test_batch_size = 10
dropout_rate = 0.2
settings_window = ''
base_dir = 'content'
src_absent_dir = []
src_present_dir = []
model = ''
src_predict_dir = []
mode = ''
width = 393
height = 393 + 128
button_font_size = 16
listbox_font_size = 9
bg_color = "black"
button_color1 = "cornflowerblue"
button_color2 = "dodgerblue4"
slider_color = "aliceblue"
window = TkinterDnD.Tk()
window.title("Binary Classifier")
window.configure(background=bg_color)
window.geometry("%dx%d+%d+%d" % (
    width, height, window.winfo_screenwidth() / 2 - width / 2, window.winfo_screenheight() / 2 - height * 11 / 20))
window.resizable(False, False)
listbox_absent = Listbox(selectmode=tkinter.SINGLE, font=("Arial", listbox_font_size, "bold"), fg="white", width=27,
                         height=8, highlightthickness=2, highlightcolor="white", bg=button_color2, borderwidth=0)
listbox_absent.insert(1, "Drop Group A directory here:")
listbox_absent.drop_target_register(DND_FILES)
listbox_absent.dnd_bind("<<Drop>>", drop_listbox_absent)
listbox_present = Listbox(selectmode=tkinter.SINGLE, font=("Arial", listbox_font_size, "bold"), fg="white", width=27,
                          height=8, highlightthickness=2, highlightcolor="white", bg=button_color2, borderwidth=0)
listbox_present.insert(1, "Drop Group B directory here:")
listbox_present.drop_target_register(DND_FILES)
listbox_present.dnd_bind("<<Drop>>", drop_listbox_present)
listbox_model = Listbox(selectmode=tkinter.SINGLE, font=("Arial", listbox_font_size, "bold"), fg="white", width=55,
                        height=4, highlightthickness=2, highlightcolor="white", bg=button_color2, borderwidth=0)
listbox_model.insert(1, "Drop model here (with file extension .h5):")
listbox_model.drop_target_register(DND_FILES)
listbox_model.dnd_bind("<<Drop>>", drop_listbox_model)
listbox_predict = Listbox(selectmode=tkinter.SINGLE, font=("Arial", listbox_font_size, "bold"), fg="white", width=55,
                          height=25, highlightthickness=2, highlightcolor="white", bg=button_color2, borderwidth=0)
listbox_predict.insert(1, "Drop files or directory to be predicted here:")
listbox_predict.drop_target_register(DND_FILES)
listbox_predict.dnd_bind("<<Drop>>", drop_listbox_predict)
listbox_console = Listbox(selectmode=tkinter.SINGLE, font=("Arial", listbox_font_size, "bold"), fg="white", width=55,
                          height=12, highlightthickness=2, highlightcolor="white", bg=button_color2, borderwidth=0)
listbox_console.insert(1, "Training progress will be shown here:")
button_create_model = Button(text="Start Training Model", font=("Arial", button_font_size, "bold"), fg="white",
                             activeforeground="white", width=29,
                             height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                             command=lambda: create_model())
button_copy_absent = Button(text="Copy Group A", font=("Arial", button_font_size, "bold"), fg="white",
                            activeforeground="white", width=14,
                            height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                            command=lambda: copy_files('absent'))
button_copy_present = Button(text="Copy Group B", font=("Arial", button_font_size, "bold"), fg="white",
                             activeforeground="white", width=14,
                             height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                             command=lambda: copy_files('present'))
button_delete = Button(text="Delete Training Files", font=("Arial", button_font_size, "bold"), fg="white",
                       activeforeground="white", width=29,
                       height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                       command=lambda: delete_files())
button_settings = Button(text="⚙", font=("Arial", button_font_size, "bold"), fg="white",
                         activeforeground="white", width=7,
                         height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                         command=lambda: open_settings())
button_switch_mode = Button(text="Switch Modes", font=("Arial", button_font_size, "bold"), fg="white",
                            activeforeground="white", width=29,
                            height=1, bg=button_color1, activebackground=button_color2, borderwidth=5, relief="raised",
                            command=lambda: switch_mode(mode))
button_switch_mode.grid(row=0, column=0)
switch_mode(mode)
window.mainloop()