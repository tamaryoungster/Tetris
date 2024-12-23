from DQN import DQN
from DQN_Agent import DQN_Agent
from Environment import *
from ReplayBuffer import ReplayBuffer
from State import State
import torch 
import wandb

epochs = 50000
C = 30
batch = 64
learning_rate = 0.1
path = "Data\DQN_PARAM_1.pth"
replay_path = "Data\Replay_1.pth"

wandb.init(
    project = "Tetris",
    id = "Tetris1",

    config={
        "name": "Tetris1",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch": batch,
        "C": C,
    }
)


def main ():
    state = State()
    env = Environment(state=state)
    player = DQN_Agent(env=env, train=True)
    player_hat = DQN_Agent(env=env,train=False)
    Q = player.DQN
    Q_hat :DQN = Q.copy()
    player_hat.DQN = Q_hat
    
    replay = ReplayBuffer()
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)

    step = 0

    for epoch in range(epochs):
        print (epoch, end="\r")
        state = env.newState()
        done = False
        while not done:
            action = player.get_Action(state, epoch=epoch)
            
            next_state, reward = env.next_state(state, action, step)
            done = env.reached_top(next_state)

            replay.push(state, action, reward, next_state, done)
            state = next_state  # env.move

            if epoch < batch:
                continue
            states, actions, rewards, next_states, dones = replay.sample(batch)
            Q_values = Q(states, actions)
            next_actions = player_hat.get_actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()

            step += 1

        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())

        if epochs%100 == 0:
            player.save_param(path)
            print(epoch, end="\r")
            wandb.log({"score": env.score})
            
            print (f'epoch: {epoch} loss: {loss:.7f}  score: {env.score}')

    player.save_param(path)
    torch.save(replay, replay_path)

if __name__ == '__main__':
    main()