import gymnasium as gym
import torch as tc

from neu_ai.RL.ActorCritic import ActorCritic


class TD_A2C(ActorCritic):
    n_opt_step = 80
    batch_size = 64

    def learn_from_memory(s):
        memory = list(map(tc.from_numpy, s.memory))
        for _ in range(s.n_opt_step):
            idx = tc.randperm(s.mem_size)[: s.batch_size]
            mem = [v[idx] for v in memory]
            obs, act, next_obs, rew, term, trunc = mem

            V, next_V = map(s.get_V, (obs, next_obs))
            logp = s.get_logp(obs, act)

            # Midbrain.VTA.DopaminergicNeurons 中脑.腹侧被盖区.多巴胺神经元:
            td_err: tc.Tensor = rew + s.gam * next_V * (1 - term) - V

            actor_loss = -(logp * td_err.detach()).mean()
            s.actor_opt.zero_grad()
            actor_loss.backward()
            s.actor_opt.step()

            critic_loss = td_err.pow(2).mean()
            s.critic_opt.zero_grad()
            critic_loss.backward()
            s.critic_opt.step()

        print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")


if __name__ == "__main__":
    algo = TD_A2C(gym.make("CartPole-v1"))
    algo.run()
