import tensorflow as tf
import traceback

print('tf', tf.__version__)
try:
    m = tf.keras.models.load_model('african_food_model.h5')
    print('loaded ok')
except Exception:
    traceback.print_exc()
