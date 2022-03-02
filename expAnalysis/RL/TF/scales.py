from tensorflow import keras

class Scale(keras.layers.Layer):
    def __init__(self, name='scale'):
        super(Scale, self).__init__(name=name)


    def build(self, inputs):

        self.factor = self.add_weight(
            name='factor',
            shape=(1),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return self.factor * inputs