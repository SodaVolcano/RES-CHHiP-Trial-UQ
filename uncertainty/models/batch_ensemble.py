"""

From uncertainty_baseline - TODO: NEED LINK AND CHECK LICENSE
"""

"""
use edward2.layers.ensemblesyncbatchnorm instead of batchnorm
use conv2DbatchEnsemble (3D?)
    x = ed.layers.Conv2DBatchEnsemble(
        filters1,
        kernel_size=1,
        use_bias=False,
        kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name=conv_name_base + '2a',
        ensemble_size=ensemble_size)(inputs)

        
"""
