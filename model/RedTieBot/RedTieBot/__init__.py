from gym.envs.registration import register

register(
    id='redtiebot-v0',
    entry_point='RedTieBot.envs:RedTieBot',
)
