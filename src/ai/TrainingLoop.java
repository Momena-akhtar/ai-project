package ai;

import java.util.ArrayList;
import java.util.List;

public class TrainingLoop {
    public static void main(String[] args) {
        // Initialize environment and agent
        int gridSize = 10;
        int[] playerPos = {0, 0};
        List<int[]> monsters = new ArrayList<>();
        monsters.add(new int[]{3, 3});
        monsters.add(new int[]{5, 5});
        List<int[]> rewards = new ArrayList<>();
        rewards.add(new int[]{2, 2});
        rewards.add(new int[]{7, 7});
        int[] goalPos = {9, 9};

        GameEnvironment env = new GameEnvironment(gridSize, playerPos, monsters, rewards, goalPos);
        DQNagent agent = new DQNagent(4, 4, 0.001, 0.99, 1.0, 0.995, 0.01);  // State size 4, action size 4

        // Training loop
        int episodes = 1000;
        for (int e = 0; e < episodes; e++) {
            int[] state = env.reset();
            int totalReward = 0;
            boolean done = false;
            while (!done) {
                int action = agent.act(state);
                GameEnvironment.Result result = env.step(action);
                int[] nextState = result.nextState;
                int reward = result.reward;
                done = result.done;
                agent.remember(state, action, reward, nextState, done);
                agent.replay(64);
                state = nextState;
                totalReward += reward;
            }

            System.out.println("Episode " + (e + 1) + "/" + episodes + ", Total Reward: " + totalReward);
        }
    }
}
