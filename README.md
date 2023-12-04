# Generative-AI based Spatio-Temporal Fusion for Video Super-Resolution through Up-Scaling and Frame-Interpolation (Ongoing Research)
## A Novel Approach leveraging Auto-Encoders, LSTM Networks and Maximum Entropy Principle.

### Objective of Study:
+ This is what the Training Data looks like :
<br> `[Grey1][Grey2,RGB2][Grey3][Grey4][Grey5,RGB5][Grey6][Grey7][Grey8,RGB8][Grey9][Grey10]`
+ Each [ ] represents Image from a moment in time.
+ The model is designed in a way to learn Temporal Dependencies between All Grey Images to be able to Generate `Grey_x` Image at Time x, enhancing Temporal Resolution. 
+ The model is designed in a way to learn Spatial Dependencies between All Grey Images having a RGB counterpart, to Generate a RGB version of Grey_x Image at Time, enhancing Spatial Resolution. 
+ The Model will be used to generate RGB counterparts of All Grey Images, so the synthetically generated dataset through Spatio-Temporal Fusion would look like:
<br> `[Grey1,RGB1][Grey2,RGB2][Grey3,RGB3][Grey4,RGB4][Grey5,RGB5][Grey6,RGB6][Grey7,RGB7][Grey8,RGB8][Grey9,RGB9][Grey10,RGB10]`


### Kindly [Review Issues](https://github.com/iSiddharth20/Spatio-Temporal-Fusion-in-Remote-Sensing/issues) Section.

### [Dataset](https://www.kaggle.com/datasets/isiddharth/spatio-temporal-data-of-moon-rise-in-raw-and-tif) is now Available!

### [Click Here](./Documentation/Concept_Presentation.pptx) for Powerpoint Presentaion of Concept.

### High Level Overview of Concept :
![System Diagram](./Documentation/System_Diagram.png)


## Thank You for Your Amazing Contribution!
