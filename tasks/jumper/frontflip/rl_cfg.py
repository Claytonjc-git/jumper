"""Starting PPO settings, inherited from the reference-guided jump recipe."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.frontflip",
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        # Zero-residual flips from HOME under action noise (64 attempts, Windows AMD CPU, native,
        # 2026-10-26): std 0.15 returns 71.7 with 1/64 crashes, std 0.6 returns -49.5 with
        # 64/64, and standing still returns 47.2. At 0.6 run 2 learned to stand; with an
        # entropy bonus its std then grew to 1.23.
        entropy_coef=0.0,
        init_std=0.15,
        # The reference commands mirror within 0.043 rad (the HOME LF_J3 offset). Run 4 without
        # this held the right claw 0.25 rad off the left and landed rolled 6 deg (same probe).
        symmetry=True,
    )
