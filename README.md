# NIPS Challenge Docker container

This repository contains the Dockerfile and setup code to run chat bot in docker instance.

## File description

- **bot.py** : Main entry point of the chat bot, message selection logic can be implemented here.
- **models/** : Folder where model code is stored
- **data/** : Folder where data files are stored
- **config.py** : Configuration script, which has the location of data files as well as bot tokens. Replace the bot token with your ones to test.
- **models/wrappper.py** - Wrapper function which calls the models. Must implement `get_response` def.
- **models/setup** - shell script to download the models
- **data/setup** - shell script to download the data files and saved model files

## Running Docker

- After installing docker, build the image from this directory using the following command: `docker build -t convai .`
- Docker will create a virtual container with all the dependencies needed.
- Docker will autostart the bot whenever the container is run: `docker run convai`

## Adding your own models

- In **models/setup**, add the repository of your model (should be a public repository for now) to clone.
- In **data/setup**, add the data location to download your saved model data
- Change the **config.py** with the endpoint of the data
- Create a wrapper in **models/wrapper.py** for your model
- Modify the **bot.py** to call your model.
- **TODO**: chat selection logic is currently sitting in bot.py. Should it be in a separate program?

## Bugs

Feel free to open an issue or submit a PR.

## Authors

Nips ConvAI Challenge McGill RLLDialog Team

