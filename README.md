# LandmarkPredictor
Used transfer learning to import ResNet50 CNN through the torch.nn modules in PyTorch framework to classify images into one of 10 landmarks. Dataset imported from the Google Landmark Recognition Challenge

Stochastic Gradient Descent through online learning (batch size left at 50), was used along with a negative log likelihood used as the loss function to minimize. 

Pandas dataframe used to maximize vectorization applications (computationally less expensive) and 

An accuracy rate of 95% was achieved when generalizing to new instances again imported from the Google Landmark Recognition Challenge.   


Landmarks: 
    1. St. Stephan’s Cathedral, Austria
    2. Teide, Spain
    3. Tallinn, Estonia
    4. Brugge, Belgium
    5. Montreal, Canada
    6. Itsukushima Shrine, Japan
    7. Shanghai, China
    8. Brisbane, Australia
    9. Edinburgh, Scotland
    10. Stockholm, Sweden
 
