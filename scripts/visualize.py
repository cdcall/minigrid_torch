import argparse
import numpy
import torch
from utils.output_utils import OutputUtils
import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000,   # 1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

# control which images are logged to tensorboard
parser.add_argument("--episode_step", type=int, default=100,
                    help="write images to tensorboard every <n> episodes")
parser.add_argument("--img_step", type=int, default=1,
                    help="write an image to tensorboard every <n> steps per candidate episode")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

tb_dir = model_dir + "/Visualize"
output_utils = OutputUtils(tb_dir, tb_dir)

# Run the agent

if args.gif:
    from array2gif import write_gif
frames = []


episode_step = 0
for episode in range(args.episodes):
    obs = env.reset()
    img_step = 0
    while True:

        # log images to tensorboard
        if episode_step % args.episode_step == 0:
            if img_step % args.img_step == 0:
                frame = numpy.moveaxis(env.render("rgb_array"), 2, 0)
                title = "agent_progress_episode_" + str(episode_step)
                output_utils.write_image_to_tensorboard(frame, title, img_step)

        img_step += 1

        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done:
            # show number of steps agent took to solve the puzzle for this episode
            output_utils.tb_writer.add_scalar("VISUALIZATION_steps_per_episode", img_step, episode_step)
            break

    episode_step += 1

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
