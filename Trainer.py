from DQN import DQN
from DQN_Agent import DQN_Agent
from Environment import *
from ReplayBuffer import ReplayBuffer
from State import State
import torch 
# import wandb

epochs = 50000
C = 30
batch = 64
learning_rate = 0.1
path = "Data\DQN_PARAM_1.pth"
replay_path = "Data\Replay_1.pth"

# wandb.init(
#     project = "Tetris",
#     id = "Tetris1",

#     config={
#         "name": "Tetris1",
#         "learning_rate": learning_rate,
#         "epochs": epochs,
#         "batch": batch,
#         "C": C,
#     }
# )


def main ():
    state = State()
    env = Environment(state=state)
    player = DQN_Agent(env=env, train=True)
    player_hat = DQN_Agent(env=env,train=False)
    Q = player.Q
    player_hat.DQN = player.DQN.copy()
    Q_hat = player_hat.Q
    
    replay = ReplayBuffer()
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)

    loss = 0
    for epoch in range(epochs):
        state = env.newState()
        done = False
        moves = 0    
        while not done:
            action = player.get_Action(state, epoch=epoch)
            moves += 1
            print(f'epoch: {epoch}   moves: {moves}', end="\r")
            next_state, reward = env.next_state(state, action)
            done = env.reached_top(next_state)

            replay.push(state, action, reward, next_state, done)
            state = next_state  # env.move

            if len(replay) < 500:
                continue
            states, actions, rewards, next_states, dones = replay.sample(batch)
            Q_values = Q(states, actions)
            next_actions, Q_hat_Values = player_hat.get_Actions_Values(next_states)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()


        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        if epoch!=0 and epoch%1 == 0:
            # player.save_param(path)
            # wandb.log({"reward": env.reward(state)})
            
            print (f'epoch: {epoch} moves: {moves} loss: {loss:.7f}  reward: {reward:.3f}')

    player.save_param(path)
    torch.save(replay, replay_path)

if __name__ == '__main__':
    main()