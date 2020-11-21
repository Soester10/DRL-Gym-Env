from tqdm import tqdm
import numpy as np

def MountainCar_Game(Neuron, env, EPISODES, min_len4train, epsilon, epsilon_mul_value, SHOW_EVERY, UPDATE_SECONDARY_WEIGHTS_NUM):
    m=0

    for episode in tqdm(range(EPISODES), ascii=True, unit='Episodes'):
        init_state = env.reset()
        done = False

        if episode % SHOW_EVERY == 0:
            render = True
            print(episode)
        else:
            render = False

        while not done:

            if np.random.random() > epsilon and episode>min_len4train:
                action = np.argmax(Neuron.nn_predicting(np.array(init_state)))
            else:
                action = np.random.randint(0, env.action_space.n)


            final_state, reward, done, _ = env.step(action)

            if final_state[0] >= env.goal_position:
                print(episode)

            if episode % SHOW_EVERY == 0:
                env.render()

            mem4training = [np.array(init_state), action, reward, np.array(final_state), done]                                #[current_state, action, reward, new_current_state, done]
        
            Neuron.updating_mem4train(mem4training)
            if episode>=min_len4train:
                Neuron.nn_training()

            if done:
                if episode % UPDATE_SECONDARY_WEIGHTS_NUM == 0:
                    Neuron.UPDATE_SECONDARY_WEIGHTS = True
                break

            init_state = final_state

        epsilon *= epsilon_mul_value


