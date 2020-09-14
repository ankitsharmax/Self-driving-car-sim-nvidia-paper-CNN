# Removing warnings
import warnings
warnings.filterwarnings('ignore')

# Removing unwanted tensorflow logs
print('Setting Up :)')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from sklearn.model_selection import train_test_split


# part 1
path = 'D:\Self-driving-car-sim-nvidia-paper-CNN'
data = importData(path)

# part 2
data = balanceData(data, display=False)



# part 3
# Data processing
imagesPath, steerings = loadData(path, data)

# part 4
# Splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(imagesPath, steerings,
                                                    test_size = 0.2,
                                                    random_state = 5)
#print('X_train size: ', len(X_train))
#print('X_test size: ', len(X_test))
#print('y_train size: ', len(y_train))
#print('y_test size: ', len(y_test))

# part 5
# Data Augmentation


# part 6
# PREPROCESSING MAIN

# part 7
# batch generation for trainig using augmentation and preprocessing
# excluding augmentation for validation/ testing images.

# part 8
# CREATE MODEL
model = createModel()
model.summary()

# part 9
# TRAINING MODEL
history = model.fit(batchGen(X_train,y_train, 100,1), steps_per_epoch = 300, epochs =25,
          validation_data = batchGen(X_test,y_test,100,0), validation_steps = 200)

# part 10
model.save('model.h5')
print('Model Saved :)')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show() 
