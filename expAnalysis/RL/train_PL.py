from tensorflow import keras

from TF.algorithm import Algorithm
from PL.models import VQC_Layer
from TF.scales import Scale
import config

def model():
    inputs = keras.Input(shape=(4,), name='input')
    hidden = VQC_Layer(inputs)
    outputs = Scale()(hidden)
    return keras.Model(inputs=[inputs], outputs=outputs)

def create_models():
    policy_model = model()
    target_model = model()

    target_model.set_weights(policy_model.get_weights())

    return policy_model, target_model

if __name__=='__main__':

    optimizer_vqc = keras.optimizers.Adam(learning_rate=config.lr_vqc)
    optimizer_cl = keras.optimizers.Adam(learning_rate=config.lr_cl)

    policy_model, target_model = create_models()

    algorithm = Algorithm(  config.env,
                            config.val_env,
                            policy_model, 
                            target_model,
                            config.replay_capacity,
                            config.epsilon_duration,
                            config.epsilon_start,
                            config.epsilon_end,
                            config.gamma,
                            optimizer_vqc,
                            optimizer_cl,
                            keras.losses.MSE,
                            config.num_steps,
                            config.update_every,
                            config.train_after,
                            config.train_every,
                            config.batch_size,
                            config.validate_every,
                            config.num_val_trials,
                            'Pennylane')

    algorithm.train()
    