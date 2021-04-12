import tensorflow.keras as keras
class SeNetBlock(object):
    def __init__(self,reduction=4):
        self.reduction = reduction
 
    def senet(self,input):
        channels = input.shape.as_list()[-1]
        avg_x = keras.layers.GlobalAveragePooling2D()(input)
        avg_x = keras.layers.Reshape((1,1,channels))(avg_x)
        avg_x = keras.layers.Conv2D(int(channels)//self.reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(avg_x)
        avg_x = keras.layers.Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(avg_x)
 
        max_x = keras.layers.GlobalMaxPooling2D()(input)
        max_x = keras.layers.Reshape((1,1,channels))(max_x)
        max_x = keras.layers.Conv2D(int(channels)//self.reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(max_x)
        max_x = keras.layers.Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(max_x)
 
        cbam_feature = keras.layers.Add()([avg_x,max_x])
 
        cbam_feature = keras.layers.Activation('hard_sigmoid')(avg_x)
 
        return keras.layers.Multiply()([input,cbam_feature])

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = inputs.shape[1]
    input_dim = inputs.shape[2]
    a = keras.layers.Permute((2, 1))(inputs)
    a = keras.layers.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = keras.layers.Permute((2, 1))(a)
    
    output_attention_mul = keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul
def build_model():
    inputdata1 = keras.Input(shape=(299, 39, 1))

    final = keras.layers.Conv2D(32, (3, 3), padding="valid", kernel_initializer='random_uniform')(inputdata1)
    final = keras.layers.PReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = keras.layers.BatchNormalization()(final)

    final = keras.layers.Conv2D(32, (3, 3), padding="valid")(final)
    final = keras.layers.PReLU()(final)
    final = keras.layers.MaxPooling2D((4, 4))(final)
    final = keras.layers.BatchNormalization()(final)

    final = keras.layers.Conv2D(64, (3, 3), padding="valid")(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = keras.layers.BatchNormalization()(final)

    #se_net = SeNetBlock()
    #final = se_net.senet(final)

    final = keras.layers.Reshape((17, 64))(final)
    final = keras.layers.GRU(64,return_sequences=True,reset_after=True)(final)
    
    final = attention_3d_block2(final)
    final = keras.layers.Flatten()(final)
    #final = keras.layers.Bidirectional(keras.layers.LSTM(64))(final)
    #final = keras.layers.Dropout(0.5)(final)
    feat1 = keras.layers.Dense(32)(final)
    
    inputdata2 = keras.Input(shape=(299, 150, 1))

    final = keras.layers.Conv2D(64, (3, 3), padding="same",activation='relu')(inputdata2)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    
    final = keras.layers.Conv2D(32, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    #final = keras.layers.Flatten()(final)
    #final = keras.layers.Dropout(0.4)(final)
    
    se_net = SeNetBlock()
    final = se_net.senet(final)
    final = keras.layers.Flatten()(final)
    #final = keras.layers.Dense(128)(final)
    feat2 = keras.layers.Dense(32)(final)
    
    final = keras.layers.Concatenate()([feat1, feat2])
    

    final = keras.layers.Dense(2)(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=[inputdata1,inputdata2], outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, decay=1e-6, epsilon=None,clipnorm=1.0)
        #,clipvalue=0.5)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    #print(model.summary())

    return model
