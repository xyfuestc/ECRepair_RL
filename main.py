import gym

from stable_baselines3 import PPO
from ECGym import ECRepairEnv

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
env = ECRepairEnv(5, 3, 10)
model = PPO.load("ppo_ec")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_ec")

del model # remove to demonstrate saving and loading


model = PPO.load("ppo_ec")

for i in range(0, 10):
    obs = env.reset()
    done = False
    step = 0
    reward = 0.0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        step += 1
        reward += rewards
        env.render()

    print('成功！总共花了{}步, reward={}.'.format(step, reward))
