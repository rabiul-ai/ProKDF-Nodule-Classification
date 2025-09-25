from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file

from keras_flops import get_flops

model_name = "EfficientNetV2B0"
n_freeze = 50
model = create_model(model_name, n_freeze)

# # build model
# inp = Input((32, 32, 3))
# x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
# x = Conv2D(64, (3, 3), activation="relu")(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)
# x = Flatten()(x)
# x = Dense(128, activation="relu")(x)
# x = Dropout(0.5)(x)
# out = Dense(10, activation="softmax")(x)
# model = Model(inp, out)

# Calculae FLOPS
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
# >>> FLOPS: 0.0338 G