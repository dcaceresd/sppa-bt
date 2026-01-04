# Learning Semantic Behavior Trees from Demonstration
This repository contains the code for learning semantic behavior trees from demonstrations.
It is based on the paper "SPPA-BT: Learning Semantic Behavior Trees from Demonstration".

**Input** -> Demonstrations (Video + trajectories)

**Output** -> SPPA-BT




## Simulation Experiments

The simulation experiments are located in the ```simulation-experiments``` folder.

## VLM prompts

The VLM prompts are located in the ```config``` folder.

## Usage
### (Optional) Docker
In a terminal, navigate to the ```docker``` folder and run the command:
```
docker build -t sppa-bt .
```

To run the container:
```
docker run --rm -it --gpus all -v /path/to/sppa-bt:/app sppa-bt
```

### Action segmentation
Action segmentation usign VLM prompts and Qwen-2.5VL. To run:

```
python src/action_segmentation.py path/to/filename.mp4
```
the output is a JSON file named ```filename.json```.

### SPPA-BT learning

```
python run.py path/to/filename.json
```
the output is a XML file named ```behavior_tree.xml```, a CSV file named ```actions_preconditions.csv``` and a folder named ```dmps``` with the dmps files.


> [!TIP]
> To visualize the generated BTs, you can use [Groot](https://www.behaviortree.dev/groot/).
