# TODO: Changeling should be the name of the net with the curriculum training function
# It uses Branches fed into ForkLayers
# forward function takes Dict[str, Tensor] and propagates through net depending on which
# branches are active, then back-props based on which branches are frozen
# should have a train(curriculum: Curriculum) function
