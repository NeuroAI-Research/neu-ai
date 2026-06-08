import numba
import numpy as np
import torch as tc
import torch.nn.functional as F

from neu_ai.nn_utils import to_np
from neu_ai.RL.ActorCritic import ActorCritic


class PPO(ActorCritic):
    n_opt_step = 80
    target_kl = 0.01
    clip_ratio = 0.2
    lam = 0.97

    def learn_from_memory(s):
        with tc.no_grad():
            # obs, act, next_obs, rew, term, trunc = s.memory
            obs, act, next_obs = map(tc.from_numpy, s.memory[:3])
            rew, term, trunc = s.memory[3:]

            V, next_V = map(s.get_V, (obs, next_obs))
            V, next_V = map(to_np, (V, next_V))
            adv, ret = ppo_adv_ret(rew, term, trunc, V, next_V, s.gam, s.lam)
            adv, ret = map(tc.from_numpy, (adv, ret))
            logp_old = s.get_logp(obs, act)

        for _ in range(s.n_opt_step):
            logp = s.get_logp(obs, act)
            kl = (logp_old - logp).mean()
            if kl > 1.5 * s.target_kl:
                break
            ratio = tc.exp(logp - logp_old)
            clip_r = tc.clamp(ratio, 1 - s.clip_ratio, 1 + s.clip_ratio)
            actor_loss = -(tc.min(ratio * adv, clip_r * adv)).mean()
            s.actor_opt.zero_grad()
            actor_loss.backward()
            s.actor_opt.step()

        for _ in range(s.n_opt_step):
            critic_loss = F.mse_loss(s.get_V(obs), ret)
            s.critic_opt.zero_grad()
            critic_loss.backward()
            s.critic_opt.step()

        print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")


@numba.njit
def ppo_adv_ret(rew, term, trunc, V, next_V, gam, lam):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        td_err = rew[t] + gam * next_V[t] * (1 - term[t]) - V[t]
        if term[t]:
            next_adv, next_ret = 0, 0
        elif t == T - 1 or trunc[t]:
            next_adv, next_ret = 0, next_V[t]
        else:
            next_adv, next_ret = adv[t + 1], ret[t + 1]
        adv[t] = td_err + gam * lam * next_adv
        ret[t] = rew[t] + gam * next_ret
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


if __name__ == "__main__":
    ppo = PPO()
    ppo.run()

"""
step_cnt: 989000, eps_ret: 1761.5294350903976
step_cnt: 990000, eps_ret: 1873.2375446039932
step_cnt: 991000, eps_ret: 1727.9603611194204
step_cnt: 992000, eps_ret: 1847.0182999400954
actor_loss: -0.0071354119514810644, critic_loss: 516.9139404296875
step_cnt: 993000, eps_ret: 1733.812706919411
step_cnt: 994000, eps_ret: 1813.2953177810239
step_cnt: 995000, eps_ret: 1916.8096961083565
step_cnt: 996000, eps_ret: 2041.0646851986794
actor_loss: -0.026656264111880802, critic_loss: 471.5614318847656
step_cnt: 997000, eps_ret: 1824.4477461229003
step_cnt: 998000, eps_ret: 1806.2452513457097
step_cnt: 999000, eps_ret: 1972.4496855301963
step_cnt: 1000000, eps_ret: 1979.1388695689243
actor_loss: -0.018384673550106485, critic_loss: 328.0265197753906
"""
