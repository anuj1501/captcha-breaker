from keras.models import load_model
model = load_model("captcha_model.hdf5")
print(model.summary())
