package ai;

import java.util.List;

public class GameEnvironment {
    private int gridSize;
    private int[] playerPos;
    private List<int[]> monsters;
    private List<int[]> rewards;
    private int[] goalPos;

    public GameEnvironment(int gridSize, int[] playerPos, List<int[]> monsters, List<int[]> rewards, int[] goalPos) {
        this.gridSize = gridSize;
        this.playerPos = playerPos;
        this.monsters = monsters;
        this.rewards = rewards;
        this.goalPos = goalPos;
    }

    public int[] reset() {
        this.playerPos = new int[]{0, 0}; // Reset player position to the start
        return getState(); // Return initial state
    }

    public int[] getState() {
        // Return a state representation as an array of integers (player position, reward positions, monster positions)
        int[] state = new int[2 + 2 * rewards.size() + 2 * monsters.size()];
        state[0] = playerPos[0];
        state[1] = playerPos[1];
        
        // Add reward positions to state
        for (int i = 0; i < rewards.size(); i++) {
            state[2 + 2 * i] = rewards.get(i)[0];
            state[3 + 2 * i] = rewards.get(i)[1];
        }
        
        // Add monster positions to state
        for (int i = 0; i < monsters.size(); i++) {
            state[2 + 2 * rewards.size() + 2 * i] = monsters.get(i)[0];
            state[3 + 2 * rewards.size() + 2 * i] = monsters.get(i)[1];
        }

        return state;
    }

    public Result step(int action) {
        // Apply action (up, down, left, right)
        switch (action) {
            case 0: playerPos[1] -= 1; break; // Move up
            case 1: playerPos[1] += 1; break; // Move down
            case 2: playerPos[0] -= 1; break; // Move left
            case 3: playerPos[0] += 1; break; // Move right
        }
        
        // Ensure player stays within bounds
        playerPos[0] = Math.max(0, Math.min(playerPos[0], gridSize - 1));
        playerPos[1] = Math.max(0, Math.min(playerPos[1], gridSize - 1));

        // Calculate the reward
        int reward = 0;
        boolean done = false;

        // Check if goal is reached
        if (playerPos[0] == goalPos[0] && playerPos[1] == goalPos[1]) {
            reward = 10;
            done = true;
        } else if (monsters.contains(playerPos)) {
            reward = -10;
            done = true;
        } else if (rewards.contains(playerPos)) {
            reward = 1;
            rewards.remove(playerPos);
        }

        return new Result(getState(), reward, done);
    }

    public static class Result {
        int[] nextState;
        int reward;
        boolean done;

        public Result(int[] nextState, int reward, boolean done) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
        }
    }
}

