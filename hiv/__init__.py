from gym.envs.registration import register

register(
    id='HIVTreatment-v0',
    entry_point='hiv.hiv:HIVTreatment',
)