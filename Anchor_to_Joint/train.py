from model_a2j import A2J
from losses import *

if __name__ == '__main__':
    input_images = None
    ground_truth_joint_pos = None
    ground_truth_joint_depth = None




    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(lr = 0.00035)

    a2j = A2J(15)

    tau1 = 1.0
    tau2 = 3.0
    losses = [smoothL1(tau1), smoothL1(tau2), smoothL1(tau1)]
    loss_weights = [3.0, 3.0, 1.0]

    a2j.compile(optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'])

    a2j.fit(x = input_images,
            y = [ground_truth_joint_pos, ground_truth_joint_depth, ground_truth_joint_pos],
            batch_size = 8,
            epochs = 34,
            verbose = 1,
            validation_split = 0.01,
            shuffle = True)
