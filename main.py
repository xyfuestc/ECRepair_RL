from stable_baselines3 import PPO
from ECGym import ECRepairEnv
# system parameter stable
import logging
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path

# env = make_vec_env("CartPole-v1", n_envs=4)


def init_logger():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    logfile = './log.txt'
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)  # 用于写到file的等级开关

    # 第三步，再创建一个handler,用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)    # 输出到console的log等级的开关

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def main():
    n, k, l = 5, 3, 3
    target_step = 5
    n_steps = 256
    total_steps = n_steps * 1000
    batch_size = 128
    env = ECRepairEnv(n, k, l)
    path = "ppo_ec_repair"+str(n)+str(k)+str(l)
    #
    # model = PPO("MlpPolicy", env, verbose=1, device="cpu", learning_rate=2e-4,
    #             gamma=0.995, gae_lambda= 0.98, n_steps=n_steps, batch_size=batch_size)

    # env = make_vec_env("ECRepairEnv-v0", n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu",
                batch_size=batch_size, n_steps=n_steps)

    # del model
    model_file = Path(path + ".zip")
    if model_file.is_file():
        print('载入model文件:{}'.format(path))
        model = PPO.load(path, env=env)
    # model.learn(total_timesteps=total_steps)
    # model.save(path)

    logger = init_logger()
    test_steps = 0
    test_nums = 0
    while True:

        # 测试一次
        obs = env.reset()
        done = False
        step = 0
        reward = 0.0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, _ = env.step(action)
            step += 1
            reward += rewards
            env.render()

        print('成功！({},{},{})总共花了{}步, reward={}.'.format(n, k, l, step, reward))
        logger.info('成功！({},{},{})总共花了{}步, reward={}.'.format(n, k, l, step, reward))

        if test_steps <= target_step:
            break

        test_nums += 1
        # 未达到要求,继续训练
        if model_file.is_file():
            model = PPO.load(path, env=env)
        model.learn(total_timesteps=total_steps*test_nums)
        model.save(path)


    print('成功达到目标要求.')
    logger.info('成功达到目标要求.')



if __name__ == '__main__':
    main()

