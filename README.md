# Overview

This is an example of how to integrate [Universal sentence encoder](https://www.google.com) (USE) to [RASA](https://rasa.com/docs/rasa/)

This example includes only 3 actions (hello, sleep, bye). The training text instance are included in `data/nlu.yml`. To add more actions or modify the chatbot, please refer to original [RASA documentation](https://rasa.com/docs/rasa/)

# Requirement

install Python 3.9.13

```
tensorflow==2.8.4
tensorflow-hub==0.12.0
rasa==3.3.3
```

# Usage

Training the model
```
rasa train
```

Test the chatbot
```
rasa shell
```