"""Starting PPO settings, inherited from the reference-guided jump recipe."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.frontflip",
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        entropy_coef=0.005,
        init_std=0.6,
        # No augmentation until a forward-rotation reference mirror is verified.
        symmetry=False,
    )
