from tensorflow.keras import backend as K

def smoothL1(tau):
    # tau = K.variable(tau)
    def loss(y_true, y_pred):
        x = K.abs(y_true - y_pred)
        x = K.switch(x < tau, K.square(x)/(2.0*tau), x - tau/2.0)
        return K.sum(x)

    return loss
