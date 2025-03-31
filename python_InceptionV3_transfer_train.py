# %% [markdown]
# # Attempt to do UK coins feature extraction by Transfer Learning from Inception v3 model by Google

# %% [markdown]
# # Clear -ONLY FOR RESETTING ENVIRONMENT
# Keras backend to try new model (Only do this if you want to change model parameters)

# %%
import tensorflow as tf
from tensorflow.keras import backend as K

# Clear the current TensorFlow graph
K.clear_session(free_memory=True)


# Reinitialize your model from scratch
model = None  # Explicitly set to None
model = None
import os, signal
os.kill(os.getpid(), signal.SIGKILL)

# %%
import os

from tensorflow.keras import layers
from tensorflow.keras import Model

# %% [markdown]
# Download Weights

# %%
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# %% [markdown]
# # Import InceptionV3

# %%
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)

# %% [markdown]
# By specifying the include_top=False argument, we load a network that doesn't include the classification layers at the top—ideal for feature extraction.

# %% [markdown]
# Let's make the model non-trainable, since we will only use it for feature extraction; we won't update the weights of the pretrained model during training.

# %%
for layer in pre_trained_model.layers:
  layer.trainable = False

# %% [markdown]
# The layer we will use for feature extraction in Inception v3 is called `mixed7`. It is not the bottleneck of the network, but we are using it to keep a sufficiently large feature map (7x7 in this case). (Using the bottleneck layer would have resulting in a 3x3 feature map, which is a bit small.) Let's get the output from `mixed7`:

# %%
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output.shape)
last_output = last_layer.output

# %% [markdown]
# Now Let's stick a fully connected layer to that:

# %%
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers 
from keras.losses import CategoricalCrossentropy

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001))(x)  # Added L2 regularization

# Add a dropout rate of 0.2
x = layers.Dropout(0.8)(x)
# Add a final Relu function for non-binary classification
x = layers.Dense(8, activation='softmax')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1),   # 'categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.0001),
              metrics=['acc'])

# %% [markdown]
# Summary of the model:

# %%
# keeping output in a small scrollable window as model is huge 
from IPython.display import HTML

def get_model_summary(model):
    # Redirect model.summary() to a string
    import io
    summary_string = io.StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    
    # Create scrollable container with monospace font for proper formatting
    return HTML(f'''
    <div style="max-height:300px; overflow:auto; font-family:monospace; white-space:pre;">
    {summary_string.getvalue()}
    </div>
    ''')

# Usage:
get_model_summary(model)

# %% [markdown]
# The model is designed and compiled
# 
# # Load our UK coins images

# %%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# !wget https://edshare.gcu.ac.uk/id/document/61325 \
#       -O /content/UK_coins_ClassSplit.zip

# local_zip = './content/UK_coins_ClassSplit.rar'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('./')
# zip_ref.close()
UK_base_dir = './UK_coins_ClassSplit'

# %% [markdown]
# Take a look at coins to confirm fully loaded:

# %%
#@title Show some coins
UK_100_dir = os.path.join(UK_base_dir, '100')
UK_050_dir = os.path.join(UK_base_dir, '050')
nrows = 4
ncols = 4
train_100_fnames = os.listdir(UK_100_dir)
train_050_fnames = os.listdir(UK_050_dir)
pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 8
next_100_pix = [os.path.join(UK_100_dir, fname)
                for fname in train_100_fnames[pic_index-8:pic_index]]
next_050_pix = [os.path.join(UK_050_dir, fname)
                for fname in train_050_fnames[pic_index-8:pic_index]]
for i, img_path in enumerate(next_100_pix+next_050_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)
  img = mpimg.imread(img_path)
  plt.imshow(img)
plt.show()

# %% [markdown]
# # Datagen
# With the same Augmentation

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255 and augmented
UK_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,        # Full rotation range for coins
    width_shift_range=0.15,    # Moderate shift
    height_shift_range=0.15,   # Moderate shift
    zoom_range=0.15,           # Slight zoom variation
    brightness_range=[0.8, 1.2], # Lighting variation
    # flipping is avoided as real case scenario a flip has the opposite coin face(heads vs tails)
    validation_split=0.2
)
B = 10 #Batch size
# Extract flow training images in batches of B images
UK_train_generator = UK_datagen.flow_from_directory(
        UK_base_dir,  # This is the source directory for training images
        target_size = (150, 150),  # All images will be resized to 150x150
        batch_size=B,
        subset = 'training',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
# Extract flow validation images in batches of B images
UK_validation_generator = UK_datagen.flow_from_directory(
        UK_base_dir,
        target_size=(150, 150),
        batch_size=B,
        subset = 'validation',
        class_mode='categorical')

# %% [markdown]
# # Train

# %%
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_acc',  # or 'val_loss' depending on what you want to track
    patience=30,  # number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore to parameters providing best val_acc
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',       # Monitor validation accuracy
    factor=0.5,              # Reduce learning rate by half when triggered
    patience=5,              # Wait 5 epochs with no improvement before reducing
    min_delta=0.01,          # Minimum change to count as improvement
    min_lr=1e-9,             # Don't reduce learning rate below this value
    verbose=1                # Print message when reducing learning rate
)

history = model.fit(
      UK_train_generator,
      steps_per_epoch=250//B,  # 250 train images = batch_size * steps
      epochs=80,
      callbacks=[early_stopping, reduce_lr], # used for restoring best epoch parameters.
      validation_data=UK_validation_generator,
      validation_steps=59//B,  # 59 validation images = batch_size * steps
      verbose=2)

# %% [markdown]
# Plot results Accuracy

# %%
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']
# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
# Get number of epochs
epochs = range(len(acc))

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training and validation accuracy on the first subplot
ax1.plot(epochs, acc, label='Training accuracy')
ax1.plot(epochs, val_acc, label='Validation accuracy')
ax1.set_title('Training and validation accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot training and validation loss on the second subplot
ax2.plot(epochs, loss, label='Training loss')
ax2.plot(epochs, val_loss, label='Validation loss')
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure
plt.show()

# %% [markdown]
# # Inference
# That should be the classifier ready. If you wish, try to play inference on a new image using model.predict
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict

# %%
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
#upload a new coin image and try to classify it
fname = "./manual_test/50p.png" #update name as needed
img = load_img(fname, target_size=(150, 150))
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
# Rescale by 1/255
x /= 255
# Let's run our image through our network
prediction = model.predict(x)
prediction
#class numbers corresponds to subdirectories (train_datagen.flow_from_directory uses alphabetic order by default)

# %%
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Path to your test folder
test_folder = "./manual_test/"

# Get list of all image files in the test folder
# Assuming common image extensions like jpg, jpeg, png
image_extensions = ['.jpg', '.jpeg', '.png']
image_files = []

for file in os.listdir(test_folder):
    # Check if the file has an image extension
    if any(file.lower().endswith(ext) for ext in image_extensions):
        image_files.append(os.path.join(test_folder, file))

# Dictionary to map coin indices to their values
# You'll need to adjust this based on your actual class indices from UK_train_generator.class_indices
# For example: {'001': 0, '002': 1, '005': 2, ...}
# Assuming class_indices values are ordered alphabetically by folder names
coin_labels = UK_train_generator.class_indices
# Invert the dictionary to map indices to labels
coin_labels_inv = {v: k for k, v in coin_labels.items()}

# Process each image and make predictions
results = []

plt.figure(figsize=(15, 15))
num_images = len(image_files)
rows = int(np.ceil(num_images / 3))  # 3 images per row

for i, image_path in enumerate(image_files):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 100))
    x = img_to_array(img)  # Numpy array with shape (100, 100, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 100, 100, 3)
    x /= 255  # Rescale by 1/255
    
    # Run prediction
    prediction = model.predict(x, verbose=0)  # Set verbose=0 to reduce output
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = coin_labels_inv[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    # Extract just the filename for display
    filename = os.path.basename(image_path)
    
    # Store results
    results.append({
        'filename': filename,
        'predicted_class': predicted_class_name,
        'confidence': confidence,
        'prediction_array': prediction[0]
    })
    
    # Display image with prediction
    plt.subplot(rows, 3, i + 1)
    plt.imshow(img)
    plt.title(f"File: {filename}\nPrediction: {predicted_class_name}\nConfidence: {confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Prediction Results:")
print("=" * 50)
for result in results:
    print(f"File: {result['filename']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("All class probabilities:")
    for class_name, index in coin_labels.items():
        prob = result['prediction_array'][index]
        print(f"  {class_name}: {prob:.4f}")
    print("-" * 50)

# Summarize overall accuracy if ground truth is available in filenames
# Assuming filenames might contain the true class (e.g., "100_coin_01.jpg" for a 100 pence coin)
# This is optional and depends on your naming convention
correct_predictions = 0
total_with_truth = 0

for result in results:
    filename = result['filename']
    # Try to extract ground truth from filename
    # This assumes filenames start with the coin value (e.g., "100_xxx.jpg")
    true_class = None
    for class_name in coin_labels.keys():
        if filename.startswith(class_name):
            true_class = class_name
            break
    
    if true_class:
        total_with_truth += 1
        if true_class == result['predicted_class']:
            correct_predictions += 1

if total_with_truth > 0:
    accuracy = correct_predictions / total_with_truth
    print(f"\nAccuracy on test set (based on filename convention): {accuracy:.2f} ({correct_predictions}/{total_with_truth})")


