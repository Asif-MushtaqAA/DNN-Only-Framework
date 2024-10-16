# DNN-Only-Framework
A pure DNN method where the flow field is predicted by the model and used to calculate aerodynamic coefficients.

In the DNN-Only method, the DNN directly predicts the flow field, which is denormalised using the original global minimum and maximum values of the flow field variables from the training dataset. From this, the aerodynamic coefficients are calculated without further refinement by CFD.  

The SDF generator and DNN UI are available in repository titled "Deep-Neural-Network".    
Link: https://github.com/Asif-MushtaqAA/Deep-Neural-Network

Note: Make sure to use correct paths for added repository.    

Example Implementation in console  
from DNN_Only import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, workflow  
workflow(58,0.6,2)  
