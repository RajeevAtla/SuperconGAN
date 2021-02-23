"""
Tests to ensure that everything is working correctly.
"""

import SuperconGAN



def test1():
    """
    Test that will instantiate a model object and fit it.
    """
    print(SuperconGAN.__version__)
    model = SuperconGAN.Synthesizer()
    model.fit()
    pass
