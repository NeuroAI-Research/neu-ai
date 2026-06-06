import gymnasium as gym

from neu_ai.rl import TD_A2C

if __name__ == "__main__":
    rl = TD_A2C(gym.make("CartPole-v1"))
    rl.run()
