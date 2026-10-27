#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array


# In[2]:


data_dir = r"D:\AIClasses\Datasets\Models\CNN\Fruits"


# In[3]:


data_dir


# In[4]:


img_height = 128
img_width = 128
batch_size = 32


# In[5]:


train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[6]:


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[7]:


class_names = train_dataset.class_names


# In[8]:


print(class_names)


# In[9]:


plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.show()


# In[10]:


for images, labels in train_dataset.take(1):

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)


# In[11]:


def rescale_images(images, labels):

    images = images / 255.0

    return images, labels


# In[12]:


train_dataset = train_dataset.map(rescale_images)


# In[13]:


validation_dataset = validation_dataset.map(rescale_images)


# In[14]:


model = tf.keras.Sequential()


# In[15]:


model.add(tf.keras.layers.Conv2D(32,(3, 3),activation="relu",input_shape=(128, 128, 3)))


# In[16]:


model.add(tf.keras.layers.MaxPooling2D())


# In[17]:


model.add(tf.keras.layers.Conv2D(64,(3, 3),activation="relu"))


# In[18]:


model.add(tf.keras.layers.MaxPooling2D())


# In[19]:


model.add(tf.keras.layers.Conv2D(128,(3, 3),activation="relu"))


# In[20]:


model.add(tf.keras.layers.MaxPooling2D())


# In[21]:


model.add(tf.keras.layers.Flatten())


# In[22]:


model.add(tf.keras.layers.Dense(128,activation="relu"))


# In[23]:


model.add(tf.keras.layers.Dense(3,activation="softmax"))


# In[24]:


model.summary()


# In[25]:


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[31]:


history = model.fit(train_dataset,validation_data=validation_dataset,epochs=10)


# In[32]:


print(history)


# In[33]:


history.history["accuracy"]


# In[34]:


history.history["val_accuracy"]


# In[35]:


plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()


# In[36]:


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


# In[37]:


model.save("fruit_classifier.keras")


# In[38]:


print("Model saved successfully!")


# In[39]:


model = tf.keras.models.load_model("fruit_classifier.keras")


# In[40]:


print("Model loaded successfully!")


# In[144]:


image_path = r"D:\AIClasses\Datasets\Models\CNN\Testimages\Car2.jpeg"


# In[145]:


banimg = load_img(image_path,target_size=(128, 128))


# In[146]:


plt.imshow(banimg )
plt.title("New Orange Image")
plt.axis("off")
plt.show()


# In[147]:


img_array = img_to_array(banimg)


# In[148]:


img_array


# In[149]:


print("Shape before batch dimension:",img_array.shape)


# In[150]:


img_array = img_array / 255.0


# In[151]:


img_array


# In[152]:


img_array = tf.expand_dims(img_array,0)


# In[153]:


print("Shape after batch dimension:",img_array.shape)


# In[154]:


predictions = model.predict(img_array)


# In[155]:


predictions[0]


# In[156]:


print("Prediction probabilities:",predictions[0])


# In[157]:


predicted_class = np.argmax(predictions[0])


# In[158]:


predicted_label = class_names[predicted_class]


# In[159]:


confidence = (np.max(predictions[0]) * 100)


# In[160]:


print("--------------------------------")
print("Predicted class:",predicted_label)
print("Confidence:",f"{confidence:.2f}%")
print("--------------------------------")


# In[161]:


plt.imshow(banimg )
plt.title(f"Prediction: {predicted_label} "f"({confidence:.2f}%)")
plt.axis("off")
plt.show()


# In[ ]:




